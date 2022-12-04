from dataclasses import dataclass, field

from sim.actors.client import Client
from sim.actors.server import Server, TrainingMode


@dataclass
class ServerInfo:
    mode: TrainingMode = field(default=TrainingMode.SYNC)
    client_ids: set[int] = field(default_factory=set)
    updates: set[int] = field(default_factory=set)


class Scheduler():
    def __init__(self) -> None:
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
        # sim = self.sim

        # server_info = self.servers[server_id]
        # for client_id in server_info.client_ids:
        #     sim.add_event(
        #         SimEvent(
        #             time=sim.now(),
        #             type=SET.SERVER_CLIENT_SYNC_START,
        #             origin=server_id,
        #             target=client_id,
        #         )
        #     )

        # # reset list of updates received
        # server_info.updates.clear()
        return

    def check_server_aggregation_readiness(self, server_id: int):
        server_info = self.servers[server_id]
        return len(server_info.updates) == len(server_info.client_ids)
