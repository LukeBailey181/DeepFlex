from dataclasses import dataclass, field

from sim.actors.client import Client
from sim.actors.server import Server, TrainingMode


@dataclass
class ServerInfo:
    mode: TrainingMode = field(default=TrainingMode.SYNC)
    client_ids: set[int] = field(default_factory=set)
    updates: set[int] = field(default_factory=set)


class Scheduler:
    def __init__(self) -> None:
        self.server_state: dict[int, ServerInfo] = {}
        self.online_clients = set()
        self.available_clients = set()

    def online_client(self, client: Client):
        self.online_clients.add(client.id)
        self.available_clients.add(client.id)

    def offline_client(self, client: Client):
        server = client.assigned_server
        server.assigned_clients.pop(client.id)
        self.server_state[server.id].client_ids.remove(client.id)

        self.available_clients.remove(client.id)
        self.online_clients.remove(client.id)

    def assign_client_to_server(self, client: Client, server: Server):
        self.server_state[server.id].client_ids.add(client.id)
        server.assigned_clients[client.id] = client
        client.assigned_server = server
        self.available_clients.remove(client.id)

    def unassign_client(self, client: Client):
        server = client.assigned_server

        self.server_state[server.id].client_ids.remove(client.id)
        server.assigned_clients.pop(client.id)
        client.assigned_server = None
        self.available_clients.add(client.id)

    def register_server(self, server: Server, mode: TrainingMode):
        self.server_state[server.id] = ServerInfo(mode=mode)

    def unregister_server(self, server: Server):
        for client_id, client in server.assigned_clients.items():
            client.assigned_server = None
            self.available_clients.add(client_id)

        self.server_state.pop(server.id)
