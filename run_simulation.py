from sim.sim import Simulation, Server, TrainingMode
from training_latency_eval.resnet_helpers import imshow
from matplotlib import pyplot as plt
import torch
from training_latency_eval.resnet_helpers import (
    get_resnet,
    get_cfar_dataset,
    RESNET_BATCH_SIZE,
)

# Number of examples, not number of batches
TRAINSET_SIZE = 64
TESTSET_SIZE = 100
BATCH_SIZE = 64
NUM_EPOCHS = 15
PRETRAINED = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_resnet_simulation():

    # Configure simulation
    simulation = Simulation()
    s1 = simulation.create_server(training_mode=TrainingMode.ASYNC)
    c1 = simulation.create_client()
    simulation.actors[c1].device = DEVICE
    server: Server = simulation.actors[s1]
    server.target_epoch = NUM_EPOCHS
    simulation.online_client(c1)

    simulation.print_actors()
    simulation.assign_client_to_server(client_id=c1, server_id=s1)
    simulation.time_limit = 800000  # Default value is 100

    # Assign dataset
    cfar_datasets = get_cfar_dataset(
        batch_size=RESNET_BATCH_SIZE,
        trainset_size=TRAINSET_SIZE,
        testset_size=TESTSET_SIZE,
    )
    train_dataset = cfar_datasets["train"]
    test_dataset = cfar_datasets["val"]
    print("Number of training minibatches:")
    print(len(train_dataset))
    server.set_train_dataset(train_dataset)
    server.set_test_dataset(test_dataset)

    # Assign model
    resnet = get_resnet(pretrained=PRETRAINED)
    server.set_model(resnet)

    # Run simulation
    simulation.run()

    # Plot losses
    losses = simulation.actors[s1].export_all_losses()

    x_vals = [i[0] for i in losses]
    y_vals = [i[1] for i in losses]

    plt.plot(x_vals, y_vals)
    plt.xlabel("Simulated time (s)")
    plt.ylabel("Traiing loss")
    plt.title("Training losses of all client models")
    plt.savefig("./loss_against_time.jpg")
    plt.clf()

    epoch_losses = simulation.actors[s1].epoch_losses
    x_vals, y_vals = [], []
    for epoch, loss in epoch_losses.items():
        x_vals.append(epoch)
        y_vals.append(loss)

    plt.plot(x_vals, y_vals)
    plt.xlabel("Epoch")
    plt.ylabel("Traiing loss")
    plt.title("Training losses of all clients per epoch")
    plt.savefig("./loss_against_epoch.jpg")
    plt.clf()

    # Print model accuracy
    acc = server.evaluate_global_model()
    print(f"Model accuracy = {acc}")

    # Plot epoch accuracy
    epoch_accs = simulation.actors[s1].epoch_accs
    x_vals, y_vals = [], []
    for epoch, acc in epoch_accs.items():
        x_vals.append(epoch)
        y_vals.append(acc)

    plt.plot(x_vals, y_vals)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Global model epoch accuracy")
    plt.savefig("./accuracy_against_epoch.jpg")
    plt.clf()

    return simulation


if __name__ == "__main__":

    run_resnet_simulation()