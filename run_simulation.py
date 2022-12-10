import csv
import os
from pathlib import Path

import torch
from icecream import ic
from matplotlib import pyplot as plt

from sim.actors import Server, TrainingMode
from sim.actors.client import Client
from sim.simulation import Simulation
from training_latency_eval.resnet_helpers import (RESNET_BATCH_SIZE,
                                                  get_cfar_dataset, get_resnet,
                                                  imshow)


def run_resnet_simulation(tmode=TrainingMode.ASYNC):
    # Number of examples, not number of batches
    TRAINSET_SIZE = 64 * 50
    TESTSET_SIZE = None
    BATCH_SIZE = 64
    NUM_EPOCHS = 1
    PRETRAINED = True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure simulation
    simulation = Simulation()
    s1 = simulation.create_server(training_mode=tmode)
    c1 = simulation.create_client(training_time=20)
    c2 = simulation.create_client(training_time=20)
    simulation.actors[c1].device = DEVICE
    simulation.actors[c2].device = DEVICE
    server: Server = simulation.actors[s1]
    server.target_epoch = NUM_EPOCHS
    simulation.online_client(c1)
    simulation.online_client(c2)

    simulation.print_actors()
    simulation.assign_client_to_server(client_id=c1, server_id=s1)
    simulation.assign_client_to_server(client_id=c2, server_id=s1)
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


def run_custom_resnet_sim(
    tmode=TrainingMode.ASYNC,
    num_clients=4,
):
    # Number of examples, not number of batches
    TRAINSET_SIZE = None
    TESTSET_SIZE = None
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    PRETRAINED = True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure simulation
    simulation = Simulation()
    s1 = simulation.create_server(training_mode=tmode)

    if num_clients == 1:
        # Baseline V100 client
        v100 = simulation.create_client(training_time=10)
        simulation.actors[v100].device = DEVICE
        simulation.online_client(v100)
        simulation.assign_client_to_server(client_id=v100, server_id=s1)

    elif num_clients == 2:
        # Baseline V100 client
        v100 = simulation.create_client(training_time=10)
        simulation.actors[v100].device = DEVICE
        simulation.online_client(v100)
        simulation.assign_client_to_server(client_id=v100, server_id=s1)

        # RTX3090 client
        rtx = simulation.create_client(training_time=19)
        simulation.actors[rtx].device = DEVICE
        simulation.online_client(rtx)
        simulation.assign_client_to_server(client_id=rtx, server_id=s1)

    elif num_clients > 2:
        # Baseline V100 clients
        for _ in range(num_clients - 1):
            v100 = simulation.create_client(training_time=10)
            simulation.actors[v100].device = DEVICE
            simulation.online_client(v100)
            simulation.assign_client_to_server(client_id=v100, server_id=s1)

        # RTX3090 client
        rtx = simulation.create_client(training_time=19)
        simulation.actors[rtx].device = DEVICE
        simulation.online_client(rtx)
        simulation.assign_client_to_server(client_id=rtx, server_id=s1)

    server: Server = simulation.actors[s1]
    server.target_epoch = NUM_EPOCHS
    simulation.print_actors()
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

    sync_str = "sync" if tmode == TrainingMode.SYNC else "async"
    config_name = f"{num_clients}client_{sync_str}"
    config_path = f"./outputs/{config_name}"
    Path(config_path).mkdir(parents=True, exist_ok=True)

    x_vals = [i[0] for i in losses]
    y_vals = [i[1] for i in losses]

    plt.plot(x_vals, y_vals)
    plt.xlabel("Simulated time (s)")
    plt.ylabel("Traiing loss")
    plt.title("Training losses of all client models")

    plt.savefig(os.path.join(config_path, "loss_against_time.png"))
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
    plt.savefig(os.path.join(config_path, "loss_against_epoch.png"))
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
    plt.savefig(os.path.join(config_path, "acc_against_epoch.png"))
    plt.clf()

    with open(os.path.join(config_path, "epoch_accs.csv"), "w") as acc_f:
        acc_f.write("time,acc\n")
        for t, v in epoch_accs.items():
            acc_f.write(f"{t},{v}\n")

    with open(os.path.join(config_path, "epoch_losses.csv"), "w") as loss_f:
        loss_f.write("time,loss\n")
        for t, v in epoch_losses.items():
            loss_f.write(f"{t},{v}\n")

    with open(os.path.join(config_path, "summary.txt"), "w") as f:
        f.write(f"Number of clients: {num_clients}\n")
        f.write(f"Training mode: {sync_str}\n")
        f.write(f"Simulation final time: {simulation.now()}\n")
        f.write(f"Model accuracy: {acc}\n")
        f.write("---\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Number of epochs: {NUM_EPOCHS}\n")

        f.writelines(actor_summary_str_arr(simulation))

    return simulation


def actor_summary_str_arr(sim: Simulation):
    summary = []
    servers = 0
    clients = 0
    for actor_id, actor in sim.actors.items():
        if actor.category == "Parameter Server":
            server: Server = actor
            summary.append(f"Server: {actor_id}\n")
            servers += 1
        else:
            client: Client = actor
            summary.append(f"Client {actor_id} - Time: {client.training_time}\n")
            clients += 1

    summary.append("---\n")
    summary.append(f"Total servers: {clients}\n")
    summary.append(f"Total clients: {clients}\n")

    return summary


if __name__ == "__main__":
    # print("async")
    # run_resnet_simulation(TrainingMode.ASYNC)
    # print("sync")
    # run_resnet_simulation(TrainingMode.SYNC)

    # num_clients_arr = [4, 8, 12]
    num_clients_arr = [1, 2, 3, 4]

    for num_clients in num_clients_arr:
        run_custom_resnet_sim(
            tmode=TrainingMode.ASYNC,
            num_clients=num_clients,
        )

        run_custom_resnet_sim(
            tmode=TrainingMode.SYNC,
            num_clients=num_clients,
        )
