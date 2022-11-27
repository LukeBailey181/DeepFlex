from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Set, Tuple
from queue import PriorityQueue
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

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

        self.assigned_server: Server = None
        self.task_complete = False
        self.model = None
        self.data = {}
        self.gradients = defaultdict(lambda: 0)
        self.staleness = 0

    def sync_model(self, global_model) -> None:
        self.model = global_model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(global_model.parameters(), lr=0.001, momentum=0.9)

    def run_training(self, batch):
        # TODO: integrate training
        inputs, labels = batch

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.update_gradients()
        pass

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
        self.assigned_clients: Set[Client] = set()
        self.client_updates = {}
        self.client_staleness_threshold = 5
        self.is_busy: bool = False

        self.current_epoch = 0
        self.target_epoch = 10

        # Dataset and iterator
        self.dataset = None
        self.dataset_iter = None
        self.server_gradient_dict = defaultdict(lambda: 0)

    # TODO: redo actions for server
    def assign_client(self, client: Client) -> None:
        client.assigned_server = self
        self.assigned_clients.add(client)

    def remove_client(self, client: Client) -> None:
        client.assigned_server = None
        self.assigned_clients.remove(client)

    def sync_model_to_client(self, client: Client):
        client.sync_model(self.global_model)

    def clear_gradients(self):
        for name in self.server_gradient_dict:
            self.server_gradient_dict[name] = 0

    def aggregate_gradients(self, gradients_list):
        for g in gradients_list:
            for name in g:
                self.server_gradient_dict[name] += g[name]

        gradients_list_length = len(gradients_list)
        for name in g:
            self.server_gradient_dict[name] /= gradients_list_length

    def set_dataset(self, dataloader: DataLoader) -> None:
        self.dataset = dataloader
        self.dataset_iter = iter(dataloader)

    def set_model(self, model) -> None:
        self.global_model = model

    def get_next_batch(self):

        try:
            batch = next(self.dataset_iter)
            return batch
        except StopIteration:
            if self.current_epoch < self.target_epoch:
                # reset iterator if target epoch not reached
                # TODO: replace this with convergence metric
                self.current_epoch += 1
                self.dataset_iter = iter(self.dataset)
                return self.get_next_batch()

        return None


@dataclass
class ServerInfo:
    mode: TrainingMode = field(default=TrainingMode.SYNC)
    client_ids: set[int] = field(default_factory=set)
    updates: set[int] = field(default_factory=set)


class Scheduler(Actor):
    def __init__(self, sim: Simulation) -> None:
        super().__init__(sim, category="Category")
        self.clients = {}
        self.servers: dict[int, ServerInfo] = {}
        self.online_clients = set()
        self.available_clients = set()

    def online_client(self, client: Client):
        self.online_clients.add(client.id)
        self.available_clients.add(client.id)

    def offline_client(self, client: Client):
        self.servers[client.assigned_server].client_ids.add(client.id)
        self.available_clients.remove(client.id)
        self.online_clients.remove(client.id)

    def assign_client_to_server(self, client: Client, server: Server):
        self.servers[server.id].client_ids.add(client.id)
        self.available_clients.remove(client.id)

    def unassign_client(self, client: Client):
        self.servers[client.assigned_server.id].client_ids.remove(client.id)
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
        server_info.updates = []

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

        self.online_clients: set[int] = set()
        self.available_clients: set[int] = set()

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
                type=SET.CLIENT_CLAIMED,
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
        ic(event)

        match event.type:

            # Simulation control events
            case SET.SIM_PAUSE:
                self.paused = True

            case SET.SIM_PRINT_ACTORS:
                self.print_actors()

            case SET.CLIENT_ONLINE:
                client: Client = self.actors[event.origin]
                self.scheduler.online_client(client)

            case SET.CLIENT_OFFLINE:
                client: Client = self.actors[event.origin]
                self.scheduler.offline_client(client)

            case SET.CLIENT_AVAILABLE:
                client: Client = self.actors[event.origin]
                self.scheduler.unassign_client(client)

            case SET.CLIENT_CLAIMED:
                # Server claims client and attempts synchronization
                server, client = self.server_client_from_event(event)

                if client.id not in self.online_clients:
                    print(
                        f"Attempting to assign offline client {client.id} to {server.id}!"
                    )

                if client.id not in self.available_clients:
                    print(
                        f"Attempting to assign unavailable client {client.id} to {server.id}!"
                    )

                server.assign_client(client)
                self.available_clients.remove(client.id)

                self.add_event(
                    SimEvent(
                        time=self.now(),
                        type=SET.SERVER_CLIENT_SYNC_START,
                        origin=server.id,
                        target=client.id,
                    )
                )

            case SET.SERVER_CLIENT_SYNC_START:
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

            case SET.SERVER_CLIENT_SYNC_END:
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

            case SET.CLIENT_TRAINING_START:
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

                client.run_training(batch)

                self.add_event(
                    SimEvent(
                        time=self.current_time + client.training_time,
                        type=SET.CLIENT_TRAINING_END,
                        origin=client.id,
                        target=server.id,
                    )
                )

            case SET.CLIENT_TRAINING_END:
                client, server = self.client_server_from_event(event)

                self.add_event(
                    SimEvent(
                        time=self.current_time,
                        type=SET.CLIENT_REQUEST_AGGREGATION,
                        origin=client.id,
                        target=server.id,
                    )
                )

            case SET.CLIENT_REQUEST_AGGREGATION:
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

            case SET.SERVER_DEFER_CLIENT_REQUEST:
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

            case SET.SERVER_CLIENT_AGGREGATION_START:
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

                gradients_list = []
                gradients_list.append(client.gradients)
                server.clear_gradients()

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

            case SET.SERVER_CLIENT_AGGREGATION_END:
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

                else:
                    if self.scheduler.servers[server.id].mode == TrainingMode.SYNC:
                        # responsible server is in sync training mode
                        if self.scheduler.check_server_aggregation_readiness(server.id):
                            # server is ready for full aggregation
                            self.scheduler.broadcast_sync_start(server.id)

                    else:
                        # responsible server is in async training mode
                        if client.id in self.online_clients:
                            # client is still online
                            self.add_event(
                                SimEvent(
                                    time=self.current_time + server.aggregation_time,
                                    type=SET.SERVER_CLIENT_SYNC_START,
                                    origin=server.id,
                                    target=client.id,
                                )
                            )

            case _:
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

            print(f"Processing event {event.type} at time {event.time}")
            self.process_event(event)

        print(f"Simulation finished running at time {self.current_time}")
