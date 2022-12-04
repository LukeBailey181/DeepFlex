from dataclasses import dataclass, field
from enum import auto, unique, Enum
from queue import PriorityQueue


@unique
class SimEventType(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self):
        return str(self.name)

    # Client availability
    CLIENT_ONLINE = auto()      # Client is newly online
    CLIENT_OFFLINE = auto()     # Client is taken offline

    CLIENT_AVAILABLE = auto()   # Client is available and can be claimed
    CLIENT_CLAIMED = auto()

    # Server activities
    SERVER_ACQUIRE_CLIENT = auto()
    SERVER_RELEASE_CLIENT = auto()

    SERVER_DEFER_CLIENT_REQUEST = auto()

    # Server receives gradients from client
    SERVER_CLIENT_AGGREGATION_START = auto()
    SERVER_CLIENT_AGGREGATION_END = auto()

    # Server processes aggregated updates
    SERVER_GLOBAL_MODEL_UPDATE_ASYNC_START = auto()
    SERVER_GLOBAL_MODEL_UPDATE_ASYNC_END = auto()

    # Server processes synchronous updates
    SERVER_GLOBAL_MODEL_UPDATE_SYNC_START = auto()
    SERVER_GLOBAL_MODEL_UPDATE_SYNC_END = auto()

    # Server sends global model to client
    SERVER_CLIENT_SYNC_START = auto()
    SERVER_CLIENT_SYNC_END = auto()

    # Client activities
    CLIENT_TRAINING_START = auto()
    CLIENT_TRAINING_END = auto()
    CLIENT_REQUEST_AGGREGATION = auto()

    # Simulation control events
    SIM_PAUSE = auto()
    SIM_PRINT_ACTORS = auto()


@dataclass(order=True, frozen=True)
class SimEvent:
    time: int
    type: SimEventType = field(compare=False)
    duration: int = field(compare=False, default=0)
    origin: int = field(compare=False, default=-1)
    target: int = field(compare=False, default=-1)


if __name__ == "__main__":
    q = PriorityQueue()

    a = SimEvent(origin=0, time=1, type=SimEventType.CLIENT_ONLINE)
    b = SimEvent(origin=0, time=3, type=SimEventType.CLIENT_ONLINE)
    c = SimEvent(origin=0, time=3, type=SimEventType.CLIENT_OFFLINE)
    d = SimEvent(origin=0, time=7, type=SimEventType.CLIENT_REQUEST_AGGREGATION)

    q.put(b)
    q.put(c)
    q.put(a)
    q.put(d)

    while not q.empty():
        event = q.get()
        print(event)
