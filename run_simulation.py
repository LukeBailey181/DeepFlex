from training_latency_eval.resnet_latency_eval import get_cfar_dataset, get_resnet_and_optimizer
from sim.sim import Simulation, Client, Server, TrainingMode
from matplotlib import pyplot as plt
from icecream import ic

# For instantiating ResNet model
RESNET_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def run_resnet_simulation():

    # Get resnet data
    # run simulation
    simulation = Simulation()

    s1 = simulation.create_server(training_mode=TrainingMode.ASYNC)
    c1 = simulation.create_client()
    server: Server = simulation.actors[s1]
    simulation.online_client(c1)
    simulation.print_actors()
    simulation.assign_client_to_server(client_id=c1, server_id=s1)
    # Default value is 100
    simulation.time_limit = 800000

    # Assign dataset
    # TODO: Handle padding, this must be multiple of batchsize 
    #cfar_data = get_cfar_dataset(trainset_size=200)
    cfar_data = get_cfar_dataset(5000)
    #cfar_data = get_cfar_dataset()
    train_dataset = cfar_data[0]['train']
    print("Number of minibatches")
    print(len(train_dataset))
    server.set_train_dataset(train_dataset)

    # Assign model
    resnet, _, _, _= get_resnet_and_optimizer(RESNET_CLASSES)
    server.set_model(resnet)

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

    return simulation

if __name__ == "__main__":

    run_resnet_simulation() 
