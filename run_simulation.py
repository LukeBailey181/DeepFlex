from training_latency_eval.resnet_latency_eval import get_cfar_dataset, get_resnet_and_optimizer
from sim.sim import Simulation, Client, Server, TrainingMode
from matplotlib import pyplot as plt
from icecream import ic

# For instantiating ResNet model
RESNET_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

TRAINSET_SIZE = 64
TESTSET_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 10


def run_resnet_simulation():

    # Configure simulation
    simulation = Simulation()
    s1 = simulation.create_server(training_mode=TrainingMode.ASYNC)
    c1 = simulation.create_client()
    server: Server = simulation.actors[s1]
    server.target_epoch = NUM_EPOCHS
    simulation.online_client(c1)
    simulation.print_actors()
    simulation.assign_client_to_server(client_id=c1, server_id=s1)
    simulation.time_limit = 800000      # Default value is 100

    # Assign dataset
    cfar_datasets = get_cfar_dataset(
        TRAINSET_SIZE,
        TESTSET_SIZE,
        BATCH_SIZE
    )[0]
    train_dataset = cfar_datasets['train']
    test_dataset = cfar_datasets['val']
    print("Number of minibatches")
    print(len(train_dataset))
    server.set_train_dataset(train_dataset)
    server.set_test_dataset(test_dataset)

    # Assign model
    resnet, _, _, _= get_resnet_and_optimizer(RESNET_CLASSES)
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
    plt.show()

    epoch_losses = simulation.actors[s1].epoch_losses
    x_vals, y_vals = [], []
    for epoch, loss in epoch_losses.items():
        x_vals.append(epoch)
        y_vals.append(loss)

    plt.plot(x_vals, y_vals)
    plt.xlabel("Epoch")
    plt.ylabel("Traiing loss")
    plt.title("Training losses of all clients per epoch")
    plt.show()

    # Print model accuracy
    acc = server.evaluate_global_model()
    print(f"Model accuracy = {acc}")

    return simulation

if __name__ == "__main__":

    run_resnet_simulation() 
