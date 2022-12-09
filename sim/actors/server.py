from collections import defaultdict
from copy import deepcopy
from enum import Enum, unique
from typing import TYPE_CHECKING, Optional

import torch
from torch import optim
from torch.utils.data import DataLoader

from sim.actors.actor import Actor

if TYPE_CHECKING:
    from sim.actors.client import Client


@unique
class TrainingMode(Enum):
    SYNC = 0
    ASYNC = 1


class Server(Actor):
    def __init__(self) -> None:
        super().__init__(category="Parameter Server")
        # TODO: make this configurable
        self.aggregation_time = 1
        self.sync_time = 1
        self.update_time = 8

        # Training mode, starts as synchronous
        self.mode: TrainingMode = TrainingMode.SYNC

        self.device = torch.device("cpu")
        self.global_model = None
        self.global_optimizer = None
        self.assigned_clients: dict[int, Client] = {}
        self.client_updates: dict[int, object] = {}
        self.client_losses = defaultdict(list)
        self.epoch_losses = defaultdict(lambda: 0)
        self.epoch_accs = {}

        self.client_staleness_threshold = 5
        self.is_busy: bool = False
        self.current_batch = False

        self.current_epoch = 0
        self.target_epoch = 10

        # Datasets
        self.train_dataset = None
        self.train_dataset_iter = None
        self.test_dataset = None

        self.server_gradient_dict = defaultdict(lambda: 0)

    def clear_updates(self):
        self.client_updates.clear()

    def clear_gradients(self):
        for name in self.server_gradient_dict:
            self.server_gradient_dict[name] = 0

    # def aggregate_gradients(self, gradients_list):
    #     for g in gradients_list:
    #         for name in g:
    #             self.server_gradient_dict[name] += g[name]

    #     # normalized gradient values
    #     gradients_list_length = len(gradients_list)
    #     for name in g:
    #         self.server_gradient_dict[name] /= gradients_list_length

    #     for name, w in self.global_model.named_parameters():
    #         self.global_model.named_parameters[name] += self.server_gradient_dict[name]

    def receive_client_update(self, client_id: int, client_model):
        client_model.to(self.device)
        model_copy = deepcopy(client_model)
        client_grads = [x for x in model_copy.parameters()]
        self.client_updates[client_id] = client_grads

    def set_train_dataset(self, dataloader: DataLoader) -> None:
        self.train_dataset = dataloader
        self.train_dataset_iter = iter(dataloader)

    def set_test_dataset(self, dataloader: DataLoader) -> None:
        self.test_dataset = dataloader

    def set_model(self, model) -> None:
        self.global_model = model
        # TODO: Don't hardcode hyperparams
        self.global_optimizer = optim.SGD(
            self.global_model.parameters(), lr=0.001, momentum=0.9
        )

    def get_next_batch(self):

        if self.current_batch % 100 == 0:
            print(f"Batch {self.current_batch}")

        try:
            batch = next(self.train_dataset_iter)
            self.current_batch += 1
            return batch
        except StopIteration:
            # end of epoch
            self.epoch_accs[self.current_epoch] = self.evaluate_global_model()
            self.current_epoch += 1
            if self.current_epoch < self.target_epoch:
                # reset iterator if target epoch not reached
                # TODO: replace this with convergence metric
                self.current_batch = 0
                self.train_dataset_iter = iter(self.train_dataset)
                return self.get_next_batch()

        return None

    def evaluate_global_model(self) -> Optional[float]:

        if self.test_dataset is None:
            return -1

        model = self.global_model
        total_correct = 0
        total_example = 0
        with torch.no_grad():
            for batch in self.test_dataset:
                inputs, labels = batch
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total_correct += (labels == preds).sum().item()
                total_example += inputs.size(0)
        model.train()

        return total_correct / total_example

    def export_all_losses(self):
        # Return list of loss tupples of form (time, loss, client_id)
        losses = []
        for client_id, client_losses in self.client_losses.items():
            for (time, loss, epoch) in client_losses:
                losses.append((time, loss, client_id))

        losses = sorted(losses, key=lambda x: x[0])

        return losses
