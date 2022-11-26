from __future__ import annotations
from enum import Enum, unique
from typing import List, Set, Tuple
from queue import PriorityQueue
from torch.utils.data import DataLoader

from icecream import ic

ic.configureOutput(includeContext=True)

from sim.events import SimEvent, SimEventType as SET


@unique
class ActorType(Enum):
    SERVER = 0
    CLIENT = 1
    SCHEDULER = 2


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
        self.task_complete = True
        self.model = {}
        self.data = {}
        self.gradients = {}

    def sync_model(self):
        pass

    def run_training(self):
        # TODO: integrate training
        pass

    def get_gradients(self):
        pass


class Server(Actor):
    def __init__(self, sim: Simulation) -> None:
        super().__init__(sim, category="Parameter Server")
        # TODO: make this configurable
        self.aggregation_time = 10
        self.sync_time = 2

        self.global_model = {}
        self.assigned_clients: Set[Client] = set()
        self.is_busy: bool = False
        # Gets assigned when experiment is run 
        self.dataset = None

    # TODO: redo actions for server
    def assign_client(self, client: Client) -> None:
        client.assigned_server = self
        self.assigned_clients.add(client)

    def remove_client(self, client: Client) -> None:
        client.assigned_server = None
        self.assigned_clients.remove(client)

    def sync_model_to_client(self, client: Client):
        client.sync_model()

    def sync_with_client(self, client: Client):
        client.sync_model()

    def set_dataset(self, dataloader: DataLoader):
        self.dataset = dataloader

class Simulation:
    def __init__(self) -> None:
        self.time_limit: int = 100
        self.current_time: int = 0
        self.event_queue: PriorityQueue[SimEvent] = PriorityQueue()
        self.timeline: PriorityQueue[SimEvent] = PriorityQueue()

        self.actor_id_counter: int = 0
        self.actors: dict[int, Actor] = {}

        self.available_clients: set[int] = set()

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

    def create_client(self, client_speed=None):
        speed = client_speed if client_speed else Client.default_speed
        new_client = Client(self, speed)
        self.actors[new_client.id] = new_client
        return new_client.id

    def create_server(self):
        new_server = Server(self)
        self.actors[new_server.id] = new_server
        return new_server.id

    def activate_client(self, client_id, time):
        if time < self.current_time:
            print(
                f"Warning! Activation time {time} is before current simulation time {self.current_time}."
            )

        self.add_event(
            time=time,
            type=SET.CLIENT_AVAILABLE,
            origin=client_id,
            target=None,
        )

    def assign_client_to_server(self, server_id: int, client_id: int) -> None:
        client: Client = self.actors[client_id]
        server: Server = self.actors[server_id]

        if client_id not in self.available_clients:
            print(
                f"Attempting to assign unavailable client {client_id} to {server_id}!"
            )
            return

        server.assign_client(client)
        self.available_clients.remove(client_id)

        self.add_event(
            time=self.current_time,
            type=SET.CLIENT_CLAIMED,
            origin=server_id,
            target=client_id,
        )

    def add_event(self, *args, event=None, **kwargs) -> SimEvent:
        """
        Use an existing event or construct a new event to add to the event queue.
        """
        e = (
            event
            if event is not None
            else SimEvent(time=kwargs.get("time"), type=kwargs.get("type"))
        )
        self.event_queue.put(e)
        self.timeline.put(e)

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
            case SET.CLIENT_ONLINE:
                self.available_clients.add(event.origin)

            case SET.CLIENT_OFFLINE:
                self.available_clients.remove(event.origin)

            case SET.CLIENT_AVAILABLE:
                self.available_clients.add(event.origin)

            case SET.CLIENT_CLAIMED:
                # Server claims client and attempts synchronization
                server, client = self.server_client_from_event(event)

                self.add_event(
                    time=self.now(),
                    type=SET.CLIENT_SYNCHRONIZE_START,
                    origin=server.id,
                    target=client.id,
                )

            case SET.CLIENT_REQUEST_AGGREGATION:
                # Client requests aggregation from server
                client, server = self.client_server_from_event(event)

                defer_event = SimEvent(
                    time=self.current_time,
                    type=SET.CLIENT_REQUEST_DEFERRED,
                    origin=server.id,
                    target=client.id,
                )

                start_event = SimEvent(
                    time=self.current_time,
                    type=SET.CLIENT_AGGREGATION_START,
                    origin=server.id,
                    target=client.id,
                )

                self.add_event(defer_event if server.is_busy else start_event)

            case SET.CLIENT_REQUEST_DEFERRED:
                # Server defers aggregation requests
                client, server = self.client_server_from_event(event)

                # TODO: make client continue training and limit number of retries
                self.add_event(
                    SimEvent(
                        # TODO: make retry time per client
                        time=self.current_time + Client.default_retry_time,
                        type=SET.CLIENT_REQUEST_AGGREGATION,
                        origin=client.id,
                        target=None,
                    )
                )

            case SET.CLIENT_AGGREGATION_START:
                client, server = self.client_server_from_event(event)
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

                self.add_event(
                    SimEvent(
                        time=self.current_time + server.aggregation_time,
                        type=SET.CLIENT_AGGREGATION_END,
                        origin=client.id,
                        target=server.id,
                    )
                )

            case SET.CLIENT_AGGREGATION_END:
                # After aggregation, make client available if training is done,
                # otherwise synchronize and continue.
                client, server = self.client_server_from_event(event)

                # TODO: 
                #       server.update_model()
                #       uses stored aggregated gradients

                if client.task_complete:
                    self.add_event(
                        SimEvent(
                            time=self.current_time,
                            type=SET.CLIENT_ONLINE,
                            origin=client.id,
                            target=server.id,
                        )
                    )

                else:
                    self.add_event(
                        SimEvent(
                            time=self.current_time + server.aggregation_time,
                            type=SET.CLIENT_SYNCHRONIZE_START,
                            origin=client.id,
                            target=server.id,
                        )
                    )

            case SET.CLIENT_SYNCHRONIZE_START:
                client, server = self.client_server_from_event(event)

                # TODO: send client updated model

                self.add_event(
                    SimEvent(
                        origin=event.origin,
                        time=self.current_time + server.sync_time,
                        type=SET.CLIENT_SYNCHRONIZE_END,
                    )
                )

            case SET.CLIENT_SYNCHRONIZE_END:
                # Immediately start training as soon as synchronization is done.
                client, server = self.client_server_from_event(event)
                server.is_busy = False

                self.add_event(
                    SimEvent(
                        origin=event.origin,
                        time=self.current_time,
                        type=SET.CLIENT_TRAINING_START,
                    )
                )

            case SET.CLIENT_TRAINING_START:
                client, server = self.client_server_from_event(event)

                # TODO: at this point, model state should be caught up, check this is the case
                # TODO: increment dataloader and send to minibatch to client
                client.run_training()

                self.add_event(
                    SimEvent(
                        origin=event.origin,
                        time=self.current_time + client.training_time,
                        type=SET.CLIENT_TRAINING_END,
                    )
                )

            case SET.CLIENT_TRAINING_END:
                self.add_event(
                    SimEvent(
                        origin=event.origin,
                        time=self.current_time,
                        type=SET.CLIENT_REQUEST_AGGREGATION,
                    )
                )

                # TODO: check if client is done training here
                # if all data is gone through, set client.task_complete = True

            case _:
                print(f"Handler for event type {event.type} not implemented!")
                ic(event)

        return

    def run(self):
        while not self.event_queue.empty() and self.current_time < self.time_limit:
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


if __name__ == "__main__":
    simulation = Simulation()
    simulation.init_clients(client_count=5, client_speeds=[1, 1, 1, 1, 1])
    simulation.init_servers(server_count=1)
    simulation.print_actors()
    simulation.assign_client_to_server(5, 0)

    simulation.run()
