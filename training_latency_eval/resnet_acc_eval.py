import torch
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
NUM_EPOCHS = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_resnet_acc():

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
    model, _ = train_resnet(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        DEVICE,
        num_epochs=NUM_EPOCHS,
    )

    # Evaluate final model accuracy on testing set
    total_correct = 0
    total_example = 0
    with torch.no_grad():
        for batch in dataloaders["val"]:
            inputs, labels = batch
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total_correct += (labels == preds).sum().item()
            total_example += inputs.size(0)

    print(f"Final testing set model acc = {total_correct/total_example}")


if __name__ == "__main__":
    collect_resnet_acc()
