from training_latency_eval.resnet_latency_eval import get_cfar_dataset, get_resnet_and_optimizer
from sim.sim import Simulation, Client, Server

# For instantiating ResNet model
RESNET_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def run_resnet_simulation():

    # Get resnet data
    # run simulation
    simulation = Simulation()

    s1 = simulation.create_server()
    c1 = simulation.create_client()
    server: Server = simulation.actors[s1]
    simulation.online_client(c1)
    simulation.print_actors()
    simulation.assign_client_to_server(client_id=c1, server_id=s1)
    # Default value is 100
    simulation.time_limit = 8000

    # Assign dataset
    # TODO: Handle padding, this must be multiple of batchsize 
    cfar_data = get_cfar_dataset(trainset_size=16)
    train_dataset = cfar_data[0]['train']
    print("DEBUG")
    print(len(train_dataset))
    server.set_dataset(train_dataset)

    # Assign model
    resnet, _, _, _= get_resnet_and_optimizer(RESNET_CLASSES)
    server.set_model(resnet)

    simulation.run()

if __name__ == "__main__":

    run_resnet_simulation() 