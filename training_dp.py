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

from training_latency_eval.resnet_helpers import (
    get_resnet,
    get_cfar_dataset,
    RESNET_BATCH_SIZE,
)

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
    batch_size_default = 256 # 1024
    learning_rate_default = 0.1
    random_seed_default = 42
    model_dir_default = "/n/holylfs05/LABS/acc_lab/Users/yujichai/hungry_hungry_ps/fasrc_scrips/models"
    model_filename_default = "resnet_distributed.pth"
    pretrained_flag = True
    resume_flag = False
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
    device_count = torch.cuda.device_count()
    print("Number of GPU:", device_count)

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
    optimizer = optim.SGD(dp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        print("Epoch: {}, Training ...".format(epoch))
        
        # Save and evaluate model routinely
        if epoch % 1 == 0:
            accuracy = evaluate(model=dp_model, device=device, test_loader=test_loader)
            #torch.save(dp_model.state_dict(), model_filepath)
            print("-" * 75)
            print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
            print("-" * 75)

        dp_model.train()

        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = dp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    
    main()