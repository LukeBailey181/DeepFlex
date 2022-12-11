from dataclasses import dataclass, field
from typing import Optional

from sim.actors.client import Client
from sim.actors.server import Server, TrainingMode


@dataclass
class ServerInfo:
    mode: TrainingMode = field(default=TrainingMode.SYNC)
    client_ids: set[int] = field(default_factory=set)
    updates: set[int] = field(default_factory=set)


@dataclass
class ClientInfo:
    assigned_server: Optional[int] = field(default=None)
    finish_time: Optional[int] = field(default=None)
    online_time: Optional[int] = field(default=None)
    offline_time: Optional[int] = field(default=None)


class Scheduler:
    def __init__(self) -> None:
        self.server_state: dict[int, ServerInfo] = {}
        self.client_state: dict[int, ClientInfo] = {}
        self.online_clients = set()
        self.available_clients = set()

    def online_client(self, client: Client):
        self.online_clients.add(client.id)
        self.available_clients.add(client.id)

    def offline_client(self, client: Client):
        self.unassign_client(client)
        self.available_clients.discard(client.id)
        self.online_clients.remove(client.id)

    def assign_client_to_server(self, client: Client, server: Server):
        self.server_state[server.id].client_ids.add(client.id)
        server.assigned_clients[client.id] = client
        client.assigned_server = server
        self.available_clients.discard(client.id)
        print(f"Assigning {client.id} to {server.id}")
        print(server.assigned_clients)

    def unassign_client(self, client: Client):
        server = client.assigned_server

        if server:
            self.server_state[server.id].client_ids.discard(client.id)
            server.assigned_clients.pop(client.id)
            print(f"Unassigning {client.id} from {server.id}")

        client.assigned_server = None
        self.available_clients.add(client.id)

    def register_server(self, server: Server, mode: TrainingMode):
        self.server_state[server.id] = ServerInfo(mode=mode)

    def unregister_server(self, server: Server):
        for client_id, client in server.assigned_clients.items():
            client.assigned_server = None
            self.available_clients.add(client_id)

        self.server_state.pop(server.id)
