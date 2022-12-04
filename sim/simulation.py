from __future__ import annotations

from queue import PriorityQueue
from typing import TYPE_CHECKING, List, Tuple

from icecream import ic

from sim.actors import Actor, Client, Server, TrainingMode
from sim.events import SimEvent
from sim.events import SimEventType as SET
from sim.scheduler import Scheduler

ic.configureOutput(includeContext=True)


class Simulation:
    def __init__(self) -> None:
        self.paused: bool = False
        self.time_limit: int = 100
        self.current_time: int = 0
        self.event_queue: PriorityQueue[SimEvent] = PriorityQueue()
        self.timeline: PriorityQueue[SimEvent] = PriorityQueue()

        self.actor_id_counter: int = 0
        self.actors: dict[int, Actor] = {}

        # scheduler handles client-server interaction
        self.scheduler = Scheduler()

    def now(self):
        return self.current_time

    def print_actors(self):
        for actor_id, actor in self.actors.items():
            print(f"Actor {actor_id}: {actor.category}")

    def init_clients(self, client_count: int, client_speeds: List[int]) -> None:
        for i in range(client_count):
            speed = client_speeds[i] if client_speeds else Client.default_speed
            new_client = Client(speed)
            self.actors[new_client.id] = new_client

    def init_servers(self, server_count: int) -> None:
        for _ in range(server_count):
            new_server = Server()
            self.actors[new_server.id] = new_server

    def create_client(self, client_speed=None, t=None):
        speed = client_speed if client_speed else Client.default_speed
        new_client = Client(speed)
        self.actors[new_client.id] = new_client
        return new_client.id

    def create_server(self, training_mode=TrainingMode.SYNC, t=None):
        new_server = Server()
        new_server.mode = training_mode
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
                type=SET.SERVER_ACQUIRE_CLIENT,
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
        # TODO: Add verbosity flag to control event printing
        #ic(event)

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

        elif event.type == SET.SERVER_ACQUIRE_CLIENT:
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
            server.is_busy = True

            # Sync global model with client
            client.sync_model(server.global_model)

            self.add_event(
                SimEvent(
                    time=self.now() + server.sync_time,
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
                    time=self.now(),
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
                        time=self.now(),
                        type=SET.CLIENT_AVAILABLE,
                        origin=client.id,
                        target=server.id,
                    )
                )
                return

            loss = client.train(batch)
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
                    time=self.now() + client.training_time,
                    type=SET.CLIENT_TRAINING_END,
                    origin=client.id,
                    target=server.id,
                )
            )

        elif event.type == SET.CLIENT_TRAINING_END:
            client, server = self.client_server_from_event(event)

            self.add_event(
                SimEvent(
                    time=self.now(),
                    type=SET.CLIENT_REQUEST_AGGREGATION,
                    origin=client.id,
                    target=server.id,
                )
            )

        elif event.type == SET.CLIENT_REQUEST_AGGREGATION:
            # Client requests aggregation from server
            client, server = self.client_server_from_event(event)

            defer_event = SimEvent(
                time=self.now(),
                type=SET.SERVER_DEFER_CLIENT_REQUEST,
                origin=server.id,
                target=client.id,
            )

            start_event = SimEvent(
                time=self.now(),
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
                        time=self.now(),
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
                        time=self.now() + 1,
                        type=SET.CLIENT_REQUEST_AGGREGATION,
                        origin=client.id,
                        target=server.id,
                    )
                )

        elif event.type == SET.SERVER_CLIENT_AGGREGATION_START:
            server, client = self.server_client_from_event(event)
            server.is_busy = True

            # TODO: keep on GPU if possible
            client.model.to("cpu")
            server.client_updates[client.id] = client.model.parameters()

            self.add_event(
                SimEvent(
                    time=self.now() + server.aggregation_time,
                    type=SET.SERVER_CLIENT_AGGREGATION_END,
                    origin=server.id,
                    target=client.id,
                )
            )

        elif event.type == SET.SERVER_CLIENT_AGGREGATION_END:
            server, client = self.server_client_from_event(event)
            server.is_busy = False

            if server.mode == TrainingMode.ASYNC:
                # immediately process async updates
                self.add_event(
                    SimEvent(
                        time=self.now(),
                        type=SET.SERVER_GLOBAL_MODEL_UPDATE_ASYNC_START,
                        origin=server.id,
                        target=client.id,
                    )
                )
                return

            if len(server.client_updates) == len(server.assigned_clients):
                # all synchronous updates received
                self.add_event(
                    SimEvent(
                        time=self.now(),  # TODO: add in update time
                        type=SET.SERVER_GLOBAL_MODEL_UPDATE_SYNC_START,
                        origin=server.id,
                        target=None,
                    )
                )

        elif event.type == SET.SERVER_GLOBAL_MODEL_UPDATE_ASYNC_START:
            server, client = self.server_client_from_event(event)
            server.is_busy = True

            param_zip = zip(
                server.client_updates[client.id], server.global_model.parameters()
            )
            for client_param, global_param in param_zip:
                global_param.grad = client_param.grad
            server.global_optimizer.step()

            self.add_event(
                SimEvent(
                    time=self.now(),  # TODO: add in update time
                    type=SET.SERVER_GLOBAL_MODEL_UPDATE_ASYNC_END,
                    origin=server.id,
                    target=client.id,
                )
            )

        elif event.type == SET.SERVER_GLOBAL_MODEL_UPDATE_ASYNC_END:
            server, client = self.server_client_from_event(event)
            server.clear_updates()
            server.is_busy = False

            # TODO: check with convergence metrics
            if client.task_complete:
                self.add_event(
                    SimEvent(
                        time=self.now(),
                        type=SET.CLIENT_AVAILABLE,
                        origin=client.id,
                        target=server.id,
                    )
                )
            else:
                self.add_event(
                    SimEvent(
                        time=self.now(),
                        type=SET.SERVER_CLIENT_SYNC_START,
                        origin=server.id,
                        target=client.id,
                    )
                )

        elif event.type == SET.SERVER_GLOBAL_MODEL_UPDATE_SYNC_START:
            server: Server = self.actors[event.origin]
            server.is_busy = True

            grad_updates = []
            for updates in server.client_updates.values():
                for idx, update in enumerate(updates):
                    if idx == 0:
                        grad_updates.append(update)
                    else:
                        grad_updates[idx] += update

            update_count = len(server.client_updates.keys())
            grad_updates = [x / update_count for x in grad_updates]

            param_zip = zip(grad_updates, server.global_model.parameters())
            for client_param, global_param in param_zip:
                global_param.grad = client_param.grad
            server.global_optimizer.step()

            self.add_event(
                SimEvent(
                    time=self.now(),
                    type=SET.SERVER_GLOBAL_MODEL_UPDATE_SYNC_END,
                    origin=server.id,
                    target=None,
                )
            )

        elif event.type == SET.SERVER_GLOBAL_MODEL_UPDATE_SYNC_END:
            server: Server = self.actors[event.origin]
            server.clear_updates()
            server.is_busy = False

            for client in server.assigned_clients.values():
                # prepare to broadcast synced model
                if client.task_complete:
                    self.add_event(
                        SimEvent(
                            time=self.now(),
                            type=SET.CLIENT_AVAILABLE,
                            origin=client.id,
                            target=server.id,
                        )
                    )
                else:
                    self.add_event(
                        SimEvent(
                            time=self.now(),
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
