from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from torch import nn, optim

from sim.actors.actor import Actor

if TYPE_CHECKING:
    from sim.actors.server import Server


class Client(Actor):
    default_speed = 1
    default_retry_time = 5

    def __init__(self, speed=1) -> None:
        super().__init__(category="Client")
        self.speed = speed
        # TODO: make this dependent on task
        self.training_time = 10 // speed

        self.assigned_server: Optional[Server]
        self.model = {}
        self.data = {}
        self.gradients = defaultdict(float)
        self.criterion = None
        self.optimizer = None

        self.task_complete: bool = False
        self.staleness = 0
        self.device = "cpu"

    def sync_model(self, global_model) -> None:
        # deepcopy to detach from global model
        self.model = deepcopy(global_model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer.zero_grad()

    def train(self, batch) -> float:
        # TODO: integrate training
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # forward + backward + optimize
        self.model.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.update_gradients()

        return loss.item()

    def clear_gradients(self):
        for name in self.gradients:
            self.gradients[name] = 0

    def update_gradients(self):
        for name, w in self.model.named_parameters():
            self.gradients[name] += w.grad
