from enum import Enum, unique


@unique
class ActorType(Enum):
    SERVER = 0
    CLIENT = 1
    SCHEDULER = 2


class Actor:
    """
    Base class for simulation actors--handles events
    """

    id_counter: int = 0

    @staticmethod
    def _assign_id():
        actor_id = Actor.id_counter
        Actor.id_counter += 1
        return actor_id

    def __init__(self, category: str = "") -> None:
        self.id = Actor._assign_id()
        self.category = category


if __name__ == "__main__":
    a1 = Actor()  # id = 0
    a2 = Actor()  # id = 1
    a3 = Actor()  # id = 2
    del a2
    a4 = Actor()  # id = 3

    print(a4.id)
