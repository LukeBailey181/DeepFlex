import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt

import time
import os
import copy
from collections import defaultdict


RESNET_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
RESNET_BATCH_SIZE = 64


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_cfar_dataset(batch_size, trainset_size, testset_size):

    # The output of torchvision datasets are PILImage images of range [0, 1].
    # Transform them to Tensors of normalized range [-1, 1].
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    if trainset_size is not None:
        trainset = [trainset[i] for i in range(trainset_size)]
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    if testset_size is not None:
        testset = [testset[i] for i in range(testset_size)]
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    dataloaders = {"train": trainloader, "val": testloader}
    # TODO remove
    # dataset_sizes = {"train": len(trainloader), "val": len(testloader)}

    return dataloaders


def get_resnet(num_classes=10, pretrained=True):

    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    # Set final dense layer size to match number of classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def get_resnet_optimizer(model, lr=0.001, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def get_resnet_scheduler(optimizer, step_size=7, gamma=0.1):
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def get_resnet_criterion():
    return nn.CrossEntropyLoss()

def train_resnet(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    num_epochs=25,
):
    model.to(device)
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    timing_data = defaultdict(list)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for minibatch_num, (inputs, labels) in enumerate(dataloaders[phase]):

                batch_start = time.time()

                if minibatch_num % 100 == 0:
                    print(f"Minibatch {minibatch_num}")

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

                batch_end = time.time()

                if phase == "train":
                    timing_data[epoch].append(batch_end - batch_start)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / len(dataloaders[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print("")

    time_elapsed = time.time() - start
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, timing_data

