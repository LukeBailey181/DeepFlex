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

    # NOTE: most events are client-focused, since we're only working with one
    # server at the moment. Once we have multiple parameter servers, we can
    # include events for interactions between servers (and possibly scheduler)

    # Client availability
    CLIENT_ONLINE = auto()  # Client is newly online
    CLIENT_OFFLINE = auto()  # Client is taken offline
    CLIENT_AVAILABLE = auto()  # Client is now available after finishing task
    CLIENT_CLAIMED = auto()  # Client is claimed by a server

    # Client aggregation and synchronization events
    CLIENT_REQUEST_AGGREGATION = auto()
    CLIENT_REQUEST_DEFERRED = auto()
    CLIENT_AGGREGATION_START = auto()
    CLIENT_AGGREGATION_END = auto()
    CLIENT_SYNCHRONIZE_START = auto()
    CLIENT_SYNCHRONIZE_END = auto()

    # Client training events
    CLIENT_TRAINING_START = auto()
    CLIENT_TRAINING_END = auto()

    # Server events
    SERVER_CLAIM_CLIENT = auto()

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
