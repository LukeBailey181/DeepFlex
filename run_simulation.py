from training_latency_eval.resnet_latency_eval import get_cfar_dataset, get_resnet_and_optimizer
from sim.sim import Simulation, Client, Server

# For instantiating ResNet model
RESNET_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def run_resnet_simulation():

    # Get resnet data
    # run simulation
    simulation = Simulation()

    simulation.init_clients(client_count=5, client_speeds=[1, 1, 1, 1, 1])
    server_id = simulation.create_server()
    server: Server = simulation.actors[server_id]
    simulation.activate_client(1, 0)
    simulation.print_actors()
    simulation.assign_client_to_server(1, server_id)

    # Assign dataset
    cfar_data = get_cfar_dataset()
    train_dataset = cfar_data[0]['train']
    server.set_dataset(train_dataset)

    # Assign model
    resnet, _, _, _= get_resnet_and_optimizer(RESNET_CLASSES)
    server.set_model(resnet)

    simulation.run()

if __name__ == "__main__":

    run_resnet_simulation() 