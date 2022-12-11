'''
Code for using only data parallel, which means it is parameter server with data parallelism. 
'''

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np
import time
import pickle

from training_latency_eval.resnet_helpers import (
    get_resnet,
    get_cfar_dataset,
    RESNET_BATCH_SIZE,
)

def analyze_list(number_list, list_name):
    number_array = np.array(number_list)
    median = np.median(number_array)
    average = np.average(number_array)
    print(list_name, ", Average:", average, ", Median:", median)

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

def main():

    num_epochs_default = 20
    batch_size_default = 64
    learning_rate_default = 0.001
    random_seed_default = 42
    model_dir_default = "/home/ec2-user/hungry_hungry_ps"
    model_filename_default = "resnet_distributed.pth"
    pretrained_flag = True
    resume_flag = False
    num_gpus = 4
    TRAINSET_SIZE = None
    TESTSET_SIZE = None

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    argv = parser.parse_args()

    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume

    if num_gpus > torch.cuda.device_count():
        raise Exception("Number of GPU is larger than available!")
    batch_size *= num_gpus
    print("Using bash size of:", batch_size)

    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''

    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Encapsulate the model on the GPU assigned to the current process
    model = torchvision.models.resnet18(pretrained=pretrained_flag)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_count = num_gpus
    print("Using", device_count, "GPU for training.")
    dp_model = torch.nn.DataParallel(model, device_ids=list(range(device_count)))
    dp_model.to(device)

    # Assign dataset
    cfar_datasets = get_cfar_dataset(
        batch_size=batch_size,
        trainset_size=TRAINSET_SIZE,
        testset_size=TESTSET_SIZE,
    )
    train_loader = cfar_datasets["train"]
    test_loader = cfar_datasets["val"]
    print("Number of training minibatches:", len(train_loader))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(dp_model.parameters(), lr=learning_rate, momentum=0.9)

    # Timer Dictionary
    time_dict = {'forward': [], 'backward': [], 'step': [], 'total': []}
    acc_list = []
    loss_list = []
    total_loss_list = []

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        print("-" * 75)
        print("Epoch: {}, Training ...".format(epoch))

        dp_model.train()
        total_loss = 0

        total_start = time.perf_counter()
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            
            # Forward Pass
            start = time.perf_counter()
            outputs = dp_model(inputs)
            dur = time.perf_counter() - start
            time_dict['forward'].append(dur)

            # Generate Loss
            loss = criterion(outputs, labels)
            loss_cpu = loss.cpu().item()
            total_loss += loss_cpu
            loss_list.append(loss_cpu)

            # Backward Pass
            start = time.perf_counter()
            loss.backward()
            dur = time.perf_counter() - start
            time_dict['backward'].append(dur)

            # Optimizer Step
            start = time.perf_counter()
            optimizer.step()
            dur = time.perf_counter() - start
            time_dict['step'].append(dur)
        total_dur = time.perf_counter() - total_start
        time_dict['total'].append(total_dur)
        print(f"Time for this epoch: {total_dur:0.4f} seconds")
        total_loss_list.append(total_loss/len(train_loader))

        dp_model.eval()
        # Save and evaluate model routinely
        if epoch % 1 == 0:
            accuracy = evaluate(model=dp_model, device=device, test_loader=test_loader)
            #torch.save(dp_model.state_dict(), model_filepath)
            acc_list.append(accuracy)
            print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))

    for list_name, number_list in time_dict.items():
        analyze_list(number_list, list_name)

    print("Accuracy List:", acc_list)
    print("Total Loss List:", total_loss_list)
    with open('./data/results/acc_list_GPU_'+str(num_gpus)+'.pickle', 'wb') as f:
        pickle.dump(acc_list, f)
    with open('./data/results/loss_list_GPU_'+str(num_gpus)+'.pickle', 'wb') as f:
        pickle.dump(loss_list, f)
    with open('./data/results/total_loss_list_GPU_'+str(num_gpus)+'.pickle', 'wb') as f:
        pickle.dump(total_loss_list, f)

if __name__ == "__main__":
    
    main()