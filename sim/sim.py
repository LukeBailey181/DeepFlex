from __future__ import annotations
import copy
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Set, Tuple, Optional
from queue import PriorityQueue
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import torch

from icecream import ic

ic.configureOutput(includeContext=True)

from .events import SimEvent, SimEventType as SET


@unique
class ActorType(Enum):
    SERVER = 0
    CLIENT = 1
    SCHEDULER = 2


@unique
class TrainingMode(Enum):
    SYNC = 0
    ASYNC = 1


class Actor:
    """
    Base class for simulation actors--handles events
    """

    def __init__(self, sim: Simulation, category: str) -> None:
        self.sim = sim
        self._assign_id()
        self.category = category

    def _assign_id(self):
        self.id = self.sim.actor_id_counter
        self.sim.actor_id_counter += 1


class Client(Actor):
    default_speed = 1
    default_retry_time = 5

    def __init__(self, sim: Simulation, speed=1) -> None:
        super().__init__(sim, category="Client")
        self.speed = speed
        # TODO: make this dependent on task
        self.training_time = 10 // speed

        self.assigned_server: Optional[Server] = None
        self.task_complete: bool = False
        self.model: Optional[torch.nn.Module] = None
        self.data = {}
        self.gradients = defaultdict(lambda: 0)
        self.staleness = 0
        self.device = "cpu"

    def sync_model(self, global_model) -> None:
        # deepcopy to detach from global model
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer.zero_grad()

    def run_training(self, batch) -> float:
        # TODO: integrate training
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.update_gradients()

        return loss.item()

    def clear_gradients(self):
        for name in self.gradients:
            self.gradients[name] = 0

    def update_gradients(self):
        for name, w in self.model.named_parameters():
            self.gradients[name] += w.grad
        pass


class Server(Actor):
    def __init__(self, sim: Simulation) -> None:
        super().__init__(sim, category="Parameter Server")
        # TODO: make this configurable
        self.aggregation_time = 10
        self.sync_time = 2

        # Training mode, starts as synchronous
        self.mode: TrainingMode = TrainingMode.SYNC

        self.global_model = None
        self.global_optimizer = None
        self.client_updates = {}
        self.client_losses = defaultdict(list)
        self.epoch_losses = defaultdict(lambda: 0)

        self.client_staleness_threshold = 5
        self.is_busy: bool = False
        self.current_batch = False

        self.current_epoch = 0
        self.target_epoch = 10

        # Datasets
        self.train_dataset = None
        self.train_dataset_iter = None
        self.test_dataset = None

        self.server_gradient_dict = defaultdict(lambda: 0)

    def sync_model_to_client(self, client: Client):
        client.sync_model(self.global_model)

    def clear_gradients(self):
        for name in self.server_gradient_dict:
            self.server_gradient_dict[name] = 0

    def aggregate_gradients(self, gradients_list):
        for g in gradients_list:
            for name in g:
                self.server_gradient_dict[name] += g[name]

        # normalized gradient values
        gradients_list_length = len(gradients_list)
        for name in g:
            self.server_gradient_dict[name] /= gradients_list_length

        for name, w in self.global_model.named_parameters():
            self.global_model.named_parameters[name] += self.server_gradient_dict[name]

    def set_train_dataset(self, dataloader: DataLoader) -> None:
        self.train_dataset = dataloader
        self.train_dataset_iter = iter(dataloader)

    def set_test_dataset(self, dataloader: DataLoader) -> None:
        self.test_dataset = dataloader

    def set_model(self, model) -> None:
        self.global_model = model
        # TODO: Don't hardcode hyperparams
        self.global_optimizer = optim.SGD(
            self.global_model.parameters(), lr=0.001, momentum=0.9
        )

    def get_next_batch(self):

        if self.current_batch % 100 == 0:
            print(f"Batch {self.current_batch}")

        try:
            batch = next(self.train_dataset_iter)
            self.current_batch += 1
            return batch
        except StopIteration:
            if self.current_epoch < self.target_epoch:
                # reset iterator if target epoch not reached
                # TODO: replace this with convergence metric
                self.current_epoch += 1
                self.current_batch = 0
                self.train_dataset_iter = iter(self.train_dataset)
                return self.get_next_batch()

        return None

    def evaluate_global_model(self) -> Optional[float]:

        if self.test_dataset is None:
            print("ABORT EVALUATION - No test dataset")
            return

        model = self.global_model
        model.eval()
        total_correct = 0
        total_example = 0
        with torch.no_grad():
            for batch in self.test_dataset:
                inputs, labels = batch
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total_correct += (labels == preds).sum().item()
                total_example += inputs.size(0)
        model.train()

        return total_correct / total_example

    def export_all_losses(self):
        # Return list of loss tupples of form (time, loss, client_id)
        losses = []
        for client_id, client_losses in self.client_losses.items():
            for (time, loss, epoch) in client_losses:
                losses.append((time, loss, client_id))

        losses = sorted(losses, key=lambda x: x[0])

        return losses


@dataclass
class ServerInfo:
    mode: TrainingMode = field(default=TrainingMode.SYNC)
    client_ids: set[int] = field(default_factory=set)
    updates: set[int] = field(default_factory=set)


class Scheduler(Actor):
    def __init__(self, sim: Simulation) -> None:
        super().__init__(sim, category="Category")
        self.servers: dict[int, ServerInfo] = {}
        self.online_clients = set()
        self.available_clients = set()

    def online_client(self, client: Client):
        self.online_clients.add(client.id)
        self.available_clients.add(client.id)

    def offline_client(self, client: Client):
        self.servers[client.assigned_server.id].client_ids.add(client.id)
        self.available_clients.remove(client.id)
        self.online_clients.remove(client.id)

    def assign_client_to_server(self, client: Client, server: Server):
        self.servers[server.id].client_ids.add(client.id)
        client.assigned_server = server
        self.available_clients.remove(client.id)

    def unassign_client(self, client: Client):
        self.servers[client.assigned_server.id].client_ids.remove(client.id)
        client.assigned_server = None
        self.available_clients.add(client.id)

    def register_server(self, server: Server, mode: TrainingMode):
        self.servers[server.id] = ServerInfo(mode=mode)

    def unregister_server(self, server_id: int):
        del self.servers[server_id]

    def broadcast_sync_start(self, server_id: int):
        sim = self.sim

        server_info = self.servers[server_id]
        for client_id in server_info.client_ids:
            sim.add_event(
                SimEvent(
                    time=sim.now(),
                    type=SET.SERVER_CLIENT_SYNC_START,
                    origin=server_id,
                    target=client_id,
                )
            )

        # reset list of updates received
        server_info.updates.clear()

    def check_server_aggregation_readiness(self, server_id: int):
        server_info = self.servers[server_id]
        return len(server_info.updates) == len(server_info.client_ids)


class Simulation:
    def __init__(self) -> None:
        self.paused: bool = False
        self.time_limit: int = 100
        self.current_time: int = 0
        self.event_queue: PriorityQueue[SimEvent] = PriorityQueue()
        self.timeline: PriorityQueue[SimEvent] = PriorityQueue()

        self.actor_id_counter: int = 0
        self.actors: dict[int, Actor] = {}

        # Schduler is the first actor in the simulation
        scheduler = Scheduler(self)
        self.scheduler = scheduler
        self.actors[scheduler.id] = scheduler

    def _validate_time(self, t: int, msg: str):
        time = t if t else self.now()
        if time < self.now():
            print(
                f"Warning! Event scheduled for time {time} before current simulation time {self.now()}."
            )
        return time

    def now(self):
        return self.current_time

    def print_actors(self):
        for actor_id, actor in self.actors.items():
            print(f"Actor {actor_id}: {actor.category}")

    def init_clients(self, client_count: int, client_speeds: List[int]) -> None:
        for i in range(client_count):
            speed = client_speeds[i] if client_speeds else Client.default_speed
            new_client = Client(self, speed)
            self.actors[new_client.id] = new_client

    def init_servers(self, server_count: int) -> None:
        for _ in range(server_count):
            new_server = Server(self)
            self.actors[new_server.id] = new_server

    def create_client(self, client_speed=None, t=None):
        speed = client_speed if client_speed else Client.default_speed
        new_client = Client(self, speed)
        self.actors[new_client.id] = new_client
        return new_client.id

    def create_server(self, training_mode=TrainingMode.SYNC, t=None):
        new_server = Server(self)
        self.actors[new_server.id] = new_server
        self.scheduler.register_server(new_server, training_mode)
        return new_server.id

    def online_client(self, client_id, t=None):
        time = t if t else self.now()

        if time < self.now():
            print(
                f"Warning! Requested activation time {time} is before current simulation time {self.now()}."
            )

        self.add_event(
            SimEvent(
                time=time,
                type=SET.CLIENT_ONLINE,
                origin=client_id,
                target=None,
            )
        )

    def offline_client(self, client_id, t=None):
        time = t if t else self.now()

        if time < self.now():
            print(
                f"Warning! Requested deactivation time {time} is before current simulation time {self.now()}."
            )

        self.add_event(
            SimEvent(
                time=time,
                type=SET.CLIENT_OFFLINE,
                origin=client_id,
                target=None,
            )
        )

    def assign_client_to_server(self, client_id: int, server_id: int, t=None) -> None:
        time = t if t else self.now()

        self.add_event(
            SimEvent(
                time=time,
                type=SET.SERVER_CLAIM_CLIENT,
                origin=server_id,
                target=client_id,
            )
        )

    def add_event(self, event: SimEvent):
        self.event_queue.put(event)
        self.timeline.put(event)

    # helpers to obtain Actor references from an event
    def client_server_from_event(self, event: SimEvent) -> Tuple[Client, Server]:
        client: Client = self.actors[event.origin]
        server: Server = self.actors[event.target]
        return (client, server)

    def server_client_from_event(self, event: SimEvent) -> Tuple[Server, Client]:
        server: Server = self.actors[event.origin]
        client: Client = self.actors[event.target]
        return (server, client)

    def server_from_event(self, event: SimEvent):
        return self.actors[event.origin]

    def process_event(self, event: SimEvent, **kwargs) -> None:
        # ic(event)

        # Simulation control events
        if event.type == SET.SIM_PAUSE:
            self.paused = True

        elif event.type == SET.SIM_PRINT_ACTORS:
            self.print_actors()

        elif event.type == SET.CLIENT_ONLINE:
            client: Client = self.actors[event.origin]
            self.scheduler.online_client(client)

        elif event.type == SET.CLIENT_OFFLINE:
            client: Client = self.actors[event.origin]
            self.scheduler.offline_client(client)

        elif event.type == SET.CLIENT_AVAILABLE:
            client: Client = self.actors[event.origin]
            self.scheduler.unassign_client(client)

        elif event.type == SET.SERVER_CLAIM_CLIENT:
            # Server claims client and attempts synchronization
            server, client = self.server_client_from_event(event)

            if client.id not in self.scheduler.online_clients:
                print(
                    f"Attempting to assign offline client {client.id} to {server.id}!"
                )

            if client.id not in self.scheduler.available_clients:
                print(
                    f"Attempting to assign unavailable client {client.id} to {server.id}!"
                )

            self.scheduler.assign_client_to_server(client, server)

            self.add_event(
                SimEvent(
                    time=self.now(),
                    type=SET.SERVER_CLIENT_SYNC_START,
                    origin=server.id,
                    target=client.id,
                )
            )

        elif event.type == SET.SERVER_CLIENT_SYNC_START:
            server, client = self.server_client_from_event(event)

            # Sync global model with client
            server.sync_model_to_client(client)

            self.add_event(
                SimEvent(
                    time=self.current_time + server.sync_time,
                    type=SET.SERVER_CLIENT_SYNC_END,
                    origin=server.id,
                    target=client.id,
                )
            )

        elif event.type == SET.SERVER_CLIENT_SYNC_END:
            # Immediately start training as soon as synchronization is done.
            server, client = self.server_client_from_event(event)
            server.is_busy = False

            self.add_event(
                SimEvent(
                    time=self.current_time,
                    type=SET.CLIENT_TRAINING_START,
                    origin=client.id,
                    target=server.id,
                )
            )

        elif event.type == SET.CLIENT_TRAINING_START:
            client, server = self.client_server_from_event(event)

            # TODO: at this point, model state should be caught up, check this is the case
            # TODO: Make sure syncing model has worked
            batch = server.get_next_batch()

            if batch is None:
                # no more data available
                self.add_event(
                    SimEvent(
                        time=self.current_time,
                        type=SET.CLIENT_AVAILABLE,
                        origin=client.id,
                        target=server.id,
                    )
                )
                return

            loss = client.run_training(batch)
            server.client_losses[client.id].append(
                [
                    self.current_time + client.training_time,
                    loss,
                    server.current_epoch,
                ]
            )
            server.epoch_losses[server.current_epoch] += loss

            self.add_event(
                SimEvent(
                    time=self.current_time + client.training_time,
                    type=SET.CLIENT_TRAINING_END,
                    origin=client.id,
                    target=server.id,
                )
            )

        elif event.type == SET.CLIENT_TRAINING_END:
            client, server = self.client_server_from_event(event)

            self.add_event(
                SimEvent(
                    time=self.current_time,
                    type=SET.CLIENT_REQUEST_AGGREGATION,
                    origin=client.id,
                    target=server.id,
                )
            )

        elif event.type == SET.CLIENT_REQUEST_AGGREGATION:
            # Client requests aggregation from server
            client, server = self.client_server_from_event(event)

            defer_event = SimEvent(
                time=self.current_time,
                type=SET.SERVER_DEFER_CLIENT_REQUEST,
                origin=server.id,
                target=client.id,
            )

            start_event = SimEvent(
                time=self.current_time,
                type=SET.SERVER_CLIENT_AGGREGATION_START,
                origin=server.id,
                target=client.id,
            )

            self.add_event(defer_event if server.is_busy else start_event)

        elif event.type == SET.SERVER_DEFER_CLIENT_REQUEST:
            # Server defers aggregation requests
            server, client = self.server_client_from_event(event)

            if client.staleness < server.client_staleness_threshold:
                # client continues training if staleness not exceeded
                self.add_event(
                    SimEvent(
                        time=self.current_time,
                        type=SET.CLIENT_TRAINING_START,
                        origin=client.id,
                        target=server.id,
                    )
                )

            else:
                # client training blocked until aggregation complete
                self.add_event(
                    SimEvent(
                        # TODO: make retry time per client
                        time=self.current_time + 1,
                        type=SET.CLIENT_REQUEST_AGGREGATION,
                        origin=client.id,
                        target=server.id,
                    )
                )

        elif event.type == SET.SERVER_CLIENT_AGGREGATION_START:
            server, client = self.server_client_from_event(event)
            server.is_busy = True

            # TODO: receive client update, perform aggregation

            # TODO: Async
            #       gradients = client.gradients
            #       server.aggregate_gradients(gradients)
            #           in the server, store aggregated gradients, this is only called once.
            #
            # TODO: Sync
            #       gradients = client.gradients
            #       server.aggregate_gradients(gradients)
            #           in sync case, wait for all clients to send updates.

            # gradients_list = []
            # gradients_list.append(client.gradients)
            # server.clear_gradients()

            # Training for async
            if self.scheduler.servers[server.id].mode == TrainingMode.ASYNC:
                client.model.to("cpu")
                param_zip = zip(
                    client.model.parameters(), server.global_model.parameters()
                )
                for client_param, global_param in param_zip:
                    global_param.grad = client_param.grad
                server.global_optimizer.step()

            server_info = self.scheduler.servers[server.id]
            if server_info.mode == TrainingMode.SYNC:
                server_info.updates.add(client.id)

            self.add_event(
                SimEvent(
                    time=self.current_time + server.aggregation_time,
                    type=SET.SERVER_CLIENT_AGGREGATION_END,
                    origin=server.id,
                    target=client.id,
                )
            )

        elif event.type == SET.SERVER_CLIENT_AGGREGATION_END:
            # After aggregation, make client available if training is done,
            # otherwise synchronize and continue.
            server, client = self.server_client_from_event(event)

            # TODO:
            #       server.update_model()
            #       uses stored aggregated gradients

            if client.task_complete:
                self.add_event(
                    SimEvent(
                        time=self.current_time,
                        type=SET.CLIENT_AVAILABLE,
                        origin=client.id,
                        target=server.id,
                    )
                )
                return

            else:
                if self.scheduler.servers[server.id].mode == TrainingMode.SYNC:
                    # responsible server is in sync training mode
                    if self.scheduler.check_server_aggregation_readiness(server.id):
                        # server is ready for full aggregation
                        self.scheduler.broadcast_sync_start(server.id)

                else:
                    # responsible server is in async training mode
                    if client.id in self.scheduler.online_clients:
                        # client is still online
                        self.add_event(
                            SimEvent(
                                time=self.current_time,
                                type=SET.SERVER_CLIENT_SYNC_START,
                                origin=server.id,
                                target=client.id,
                            )
                        )

        else:
            print(f"Handler for event type {event.type} not implemented!")
            ic(event)

        return

    def run(self):
        self.paused = False

        while (
            not self.paused
            and not self.event_queue.empty()
            and self.current_time < self.time_limit
        ):
            event = self.event_queue.get()

            # report late events--this should never happen
            if event.time < self.current_time:
                print(
                    f"Warning! Late event: current time is {self.current_time} but event was scheduled for {event.time}"
                )
                ic(event)
                ic(self.event_queue)

            # update to next time
            self.current_time = max(self.current_time, event.time)

            # print(f"Processing event {event.type} at time {event.time}")
            self.process_event(event)

        print(f"Simulation finished running at time {self.current_time}")
