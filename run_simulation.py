import csv
import math
import os
import random
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
    tmode=TrainingMode.SYNC,
    num_clients=4,
    client_train_time=10,
    server_update_time=2,
    uniform=False,
):
    # Number of examples, not number of batches
    TRAINSET_SIZE = None
    TESTSET_SIZE = None
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    PRETRAINED = True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure simulation
    simulation = Simulation()
    s1 = simulation.create_server(training_mode=tmode)

    if uniform:
        for _ in range(num_clients):
            c = simulation.create_client(training_time=10)
            c_actor: Client = simulation.actors[c]
            c_actor.device = DEVICE
            c_actor.training_time = client_train_time
            simulation.online_client(c)
            simulation.assign_client_to_server(client_id=c, server_id=s1)

    else:
        if num_clients == 1:
            # Baseline V100 client
            v100 = simulation.create_client(training_time=10)
            simulation.actors[v100].device = DEVICE
            simulation.online_client(v100, t=0)
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
            for x in range(num_clients - 1):
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
    server.update_time = server_update_time
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


def run_dynamic_env_sim(tmode=TrainingMode.ASYNC, arrival_times=[0], duration=None):
    # Number of examples, not number of batches
    TRAINSET_SIZE = 32000
    TESTSET_SIZE = None
    BATCH_SIZE = 64
    NUM_EPOCHS = 1
    PRETRAINED = True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure simulation
    simulation = Simulation()
    s1 = simulation.create_server(training_mode=tmode)

    # Baseline V100 client
    v100 = simulation.create_client(training_time=10)
    simulation.actors[v100].device = DEVICE
    simulation.online_client(v100, t=0)
    simulation.assign_client_to_server(client_id=v100, server_id=s1)

    # additional clients
    v100_2 = simulation.create_client(training_time=10)
    rtx3090_0 = simulation.create_client(training_time=19)
    rtx3090_1 = simulation.create_client(training_time=19)

    # schedule onlining and assignment of clients
    simulation.online_client(v100_2, t=arrival_times[1])
    simulation.online_client(rtx3090_0, t=arrival_times[2])
    simulation.online_client(rtx3090_1, t=arrival_times[3])
    simulation.assign_client_to_server(
        client_id=v100_2, server_id=s1, t=arrival_times[1]
    )
    simulation.assign_client_to_server(
        client_id=rtx3090_0, server_id=s1, t=arrival_times[2]
    )
    simulation.assign_client_to_server(
        client_id=rtx3090_1, server_id=s1, t=arrival_times[3]
    )

    # schedule offlining of clients
    if duration is not None:
        simulation.offline_client(v100_2, t=arrival_times[1] + duration)
        simulation.offline_client(rtx3090_0, t=arrival_times[2] + duration)
        simulation.offline_client(rtx3090_1, t=arrival_times[3] + duration)

    # set client devices
    simulation.actors[v100].device = DEVICE
    simulation.actors[v100_2].device = DEVICE
    simulation.actors[rtx3090_0].device = DEVICE
    simulation.actors[rtx3090_1].device = DEVICE

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
    config_name = f"dynclient_{sync_str}"
    config_path = f"./dyn_outputs/{config_name}"
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
        f.write(f"Number of clients: 4\n")
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


def get_arrival_times(rate_param: float, num_events: int) -> list[int]:
    times = []

    while len(times) < num_events:
        next_time = round(-math.log(1.0 - random.random()) / rate_param, 0)
        times.append(int(next_time))

    return times


if __name__ == "__main__":
    # print("async")
    # run_resnet_simulation(TrainingMode.ASYNC)
    # print("sync")
    # run_resnet_simulation(TrainingMode.SYNC)

    # average rate of arrival is 1 client every 5 seconds = 1 / (5 * 1000)
    print(sorted(get_arrival_times(1 / (5 * 1000), 5)))

    # num_clients_arr = [4, 8, 12]
    # num_clients_arr = [1, 2, 3, 4]

    # for num_clients in num_clients_arr:
    #     run_custom_resnet_sim(
    #         tmode=TrainingMode.ASYNC,
    #         num_clients=num_clients,
    #     )

    #     run_custom_resnet_sim(
    #         tmode=TrainingMode.SYNC,
    #         num_clients=num_clients,
    #     )

    run_custom_resnet_sim(
        tmode=TrainingMode.SYNC,
        num_clients=4,
        client_train_time=10,
        server_update_time=2,
        uniform=True,
    )

    # dynamic experiments
    # r_param = 1 / (1 * 1000)
    # arrivals = [0] + get_arrival_times(r_param, 3)
    # duration = 10 * 100

    # print(arrivals)

    # run_dynamic_env_sim(
    #     tmode=TrainingMode.ASYNC, arrival_times=arrivals, duration=duration
    # )

    # dynamic environment experiments
