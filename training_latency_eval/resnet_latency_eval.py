import torch
import numpy as np
from matplotlib import pyplot as plt
from resnet_helpers import (
    get_resnet,
    get_resnet_optimizer,
    get_resnet_criterion,
    get_resnet_scheduler,
    get_cfar_dataset,
    train_resnet,
    RESNET_BATCH_SIZE,
)

TRAINSET_SIZE = None
TESTSET_SIZE = None
PRETAINED = True
NUM_EPOCHS = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DROP_FIRST = 3


def collect_resnet_latency():

    # Get datasset
    dataloaders = get_cfar_dataset(
        batch_size=RESNET_BATCH_SIZE,
        trainset_size=TRAINSET_SIZE,
        testset_size=TESTSET_SIZE,
    )

    print(f"Number of batches in training dataset : {len(dataloaders['train'])}")

    model = get_resnet(pretrained=PRETAINED)
    optimizer = get_resnet_optimizer(model)
    scheduler = get_resnet_scheduler(optimizer)
    criterion = get_resnet_criterion()

    # Train and time
    print("Training ResNet")
    model, timing_data = train_resnet(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        DEVICE,
        num_epochs=NUM_EPOCHS,
    )

    epoch_0 = timing_data[0]
    plt.ylabel("Minibatch latency (s)")
    plt.xlabel("Minibatch number")
    plt.title("ResNet-18 training latency")
    plt.plot([i for i in range(len(epoch_0))], epoch_0)
    plt.savefig("ResNet-18_latency.png")

    print(f"Number of timing datapoints = {len(epoch_0[DROP_FIRST:])}")
    print(f"Mean minibatch training time = {np.array(epoch_0)[DROP_FIRST:].mean()}")


if __name__ == "__main__":
    collect_resnet_latency()
