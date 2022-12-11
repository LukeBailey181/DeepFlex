from dataclasses import dataclass, field
from enum import auto, unique, Enum
from queue import PriorityQueue


@unique
class SimEventType(Enum):
    @staticmethod
    def _generate_next_value_(name, *_):
        return name

    def __str__(self):
        return str(self.name)

    # Client availability
    CLIENT_ONLINE = auto()      # Client is newly online
    CLIENT_OFFLINE = auto()     # Client is taken offline

    CLIENT_AVAILABLE = auto()   # Client is available and can be claimed
    CLIENT_CLAIMED = auto()

    # Client should finish current task and go offline
    NOTIFY_CLIENT_OFFLINE = auto()

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
    SERVER_SEND_MODEL_START = auto()
    SERVER_SEND_MODEL_END = auto()

    # Client activities
    CLIENT_TRAINING_START = auto()
    CLIENT_TRAINING_END = auto()
    CLIENT_REQUEST_AGGREGATION = auto()

    # Simulation control events
    SIM_PAUSE = auto()
    SIM_PRINT_ACTORS = auto()


SET = SimEventType


SET_EVENT_PRIORITIES = {
    # Priorty 0: Simulation control events must be processed first
    SET.SIM_PAUSE: 0,
    SET.SIM_PRINT_ACTORS: 0,
    # Priorty 1: Client/Server online status
    SET.CLIENT_ONLINE: 1,
    SET.CLIENT_OFFLINE: 1,
    # Priority 2: Client/Server availability
    SET.CLIENT_AVAILABLE: 2,
    SET.CLIENT_CLAIMED: 2,
    SET.SERVER_ACQUIRE_CLIENT: 2,
    SET.SERVER_RELEASE_CLIENT: 2,
    SET.NOTIFY_CLIENT_OFFLINE: 2,
    # Priority 3: _END events processed first before _START events
    SET.SERVER_CLIENT_AGGREGATION_END: 3,
    SET.SERVER_GLOBAL_MODEL_UPDATE_ASYNC_END: 3,
    SET.SERVER_GLOBAL_MODEL_UPDATE_SYNC_END: 3,
    SET.SERVER_SEND_MODEL_END: 3,
    SET.CLIENT_TRAINING_END: 3,
    # Priority 4: _START events
    SET.SERVER_CLIENT_AGGREGATION_START: 4,
    SET.SERVER_GLOBAL_MODEL_UPDATE_ASYNC_START: 4,
    SET.SERVER_GLOBAL_MODEL_UPDATE_SYNC_START: 4,
    SET.SERVER_SEND_MODEL_START: 4,
    SET.CLIENT_TRAINING_START: 4,
    # Aggregation requests and deferrals are considered "start" events
    SET.CLIENT_REQUEST_AGGREGATION: 4,
    SET.SERVER_DEFER_CLIENT_REQUEST: 4,
}


@dataclass(order=True, frozen=True)
class SimEvent:
    time: int = field(compare=True)
    type: SimEventType = field(compare=False)
    priority: int = field(compare=True, init=False)
    origin: int = field(compare=False, default=-1)
    target: int = field(compare=False, default=-1)
    duration: int = field(compare=False, default=0)

    def __post_init__(self):
        object.__setattr__(self, "priority", SET_EVENT_PRIORITIES[self.type])


if __name__ == "__main__":
    missing_event_priorities = []
    for event_type in SET:
        if event_type not in SET_EVENT_PRIORITIES:
            missing_event_priorities.append(event_type)

    print("Events missing priorities:")
    for event_type in missing_event_priorities:
        print(event_type)

    # Event priorty test
    q = PriorityQueue()

    q.put(SimEvent(origin=1, time=8, type=SimEventType.CLIENT_TRAINING_START))
    q.put(SimEvent(origin=2, time=8, type=SimEventType.CLIENT_TRAINING_START))
    q.put(SimEvent(origin=0, time=8, type=SimEventType.CLIENT_TRAINING_END))
    q.put(SimEvent(origin=0, time=5, type=SimEventType.CLIENT_TRAINING_START))
    q.put(SimEvent(origin=1, time=8, type=SimEventType.CLIENT_TRAINING_END))
    q.put(SimEvent(origin=1, time=3, type=SimEventType.CLIENT_ONLINE))
    q.put(SimEvent(origin=0, time=1, type=SimEventType.CLIENT_ONLINE))
    q.put(SimEvent(origin=0, time=8, type=SimEventType.CLIENT_OFFLINE))

    while not q.empty():
        event = q.get()
        print(event)
