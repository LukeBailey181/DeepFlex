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

#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
#DEVICE = torch.device("mps")
RESNET_BATCH_SIZE = 64
DATASET_SIZE = None
EPOCHS = 10

def train_resnet(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    dataset_sizes,
    num_epochs=25,
):
    model.to(device)

    since = time.time()

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

                start = time.time()

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
                print("debug")
                print(torch.sum(preds == labels.data).item())
                print(torch.sum(preds == labels.data))

                end = time.time()

                if phase == "train":
                    timing_data[epoch].append(end - start)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val":
                print(epoch_acc)
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, timing_data


def get_cfar_dataset(
    trainset_size=None, testset_size=None, batch_size=RESNET_BATCH_SIZE
):

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
    dataset_sizes = {"train": len(trainloader), "val": len(testloader)}

    classes = (
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

    return dataloaders, dataset_sizes, classes


def get_resnet_and_optimizer(classes):

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(classes))

    model_ft = model_ft.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler


def collect_timing_data_resnet():

    # Get datasset
    print("Collecting CFAR dataset")
    dataloaders, dataset_sizes, classes = get_cfar_dataset(DATASET_SIZE)

    print(len(dataloaders["train"]))

    # Get model and training infrastructure
    print("Instantiating resnet")
    model, criterion, optimizer, scheduler = get_resnet_and_optimizer(classes)

    # Train and time
    print("Training ResNet")
    model, timing_data = train_resnet(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        DEVICE,
        dataset_sizes,
        num_epochs=EPOCHS,
    )

    return model, timing_data


def evaluate_resnet():

    model, timing_data = collect_timing_data_resnet()

    epoch_0 = timing_data[0]
    plt.ylabel("Minibatch latency (s)")
    plt.xlabel("Minibatch number")
    plt.title("ResNet-18 training latency")
    plt.plot([i for i in range(len(epoch_0))], epoch_0)
    plt.savefig("ResNet-18_latency.png")

    print(f"\nRESNET TRAINING TIME DATAPOINTS = {len(epoch_0[2:])}")
    print(f"\nRESNET MEAN TRAINING TIME VALUE = {np.array(epoch_0)[10:].mean()}")


if __name__ == "__main__":
    evaluate_resnet()
