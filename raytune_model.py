import os
import time
import random
import gc
import multiprocessing
from collections import Counter

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torchvision
from torchvision import datasets, models, transforms

import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader, Subset, SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from shutil import copyfile

import pandas as pd
from pandas import DataFrame
import seaborn as sns

from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

torch.multiprocessing.set_sharing_strategy('file_system')

use_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
if device.type == 'cuda':
  print('CUDA is available. Training on GPU')
else:
  print('CUDA is not available. Training on CPU')

train_data_dir = "/home/nagianek/projects/aps360/aps360_project_db/DATASET/TRAIN"
test_data_dir = "/home/nagianek/projects/aps360/aps360_project_db/DATASET/TEST"

# Source: https://github.com/ufoym/imbalanced-dataset-sampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))             if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)             if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def create_dataset(train=True):
    """ Load data from file system and create a dataset
    Args:
        train: choose between training set and test set
    Returns:
        dataset 
    """
    # convert images to (224 x 224)
    data_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ColorJitter(brightness=0.1, hue=0.05, saturation=0.01),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(35),
                                        transforms.ToTensor()])

    # load dataset
    if train:
        return datasets.ImageFolder(train_data_dir, transform=data_transform)  
    else:
        return datasets.ImageFolder(test_data_dir, transform=data_transform)
  
def split_data(dataset, train=True, train_ratio=0.7, batch_size=25, 
                                                    seed=1000, shuffle=True):
    """ Load data from dataset and split it into train/valid/test sets
    Args:
        dataset: dataset containing all data
        train: determines if dataset is for the train or the test set. 
                If it's the test set, don't split
        train_ratio: percentage of data to go into training set
        batch_size: batch size for dataloader
        seed: random seed
        shuffle: flag on whether the shuffle original dataset or not
    Returns:
        train_loader: training dataloader
        valid_loader: validation dataloader
        test_loader: test dataloader    
    """
    # number of examples in dataset
    num_data = len(dataset)

    # shuffle data
    if shuffle:
        torch.manual_seed(seed)
        indices = torch.randperm(num_data)
    else:
        indices = torch.arange(num_data)

    # dataloader parameters
    num_workers = multiprocessing.cpu_count()
    prefetch_factor = 8
    pin_mem = False
    if torch.cuda.is_available():
        pin_mem = True

    if train:
        # calculate indices to split data at
        split = int(np.floor(train_ratio * num_data))

        # train/valid split
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = Subset(dataset, train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # create Data loaders
        train_loader = DataLoader(dataset, 
                                  batch_size=batch_size, 
                                  sampler=ImbalancedDatasetSampler(train_sampler), 
                                  num_workers=num_workers, 
                                  pin_memory=pin_mem,
                                  prefetch_factor=prefetch_factor)
        
        valid_loader = DataLoader(dataset, 
                                  batch_size=batch_size, 
                                  sampler=valid_sampler, 
                                  num_workers=num_workers, 
                                  pin_memory=pin_mem,
                                  prefetch_factor=prefetch_factor)
        
        return train_loader, valid_loader

    else:
        # all data placed in test set
        test_idx = indices
        test_sampler = SubsetRandomSampler(test_idx)

        test_loader = DataLoader(dataset, 
                                 batch_size=batch_size, 
                                 sampler=test_sampler, 
                                 num_workers=num_workers, 
                                 pin_memory=pin_mem,
                                 prefetch_factor=prefetch_factor)
        
        return test_loader

def get_num_params(net):
    """determine the size of the model
    Args:
        net: machine learning model
    Returns:
        size of model
    """
    return sum(p.numel() for p in net.parameters())

class CNN(nn.Module):
    def __init__(self, nin: int, nout: int, k=5, s=1):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=k, stride=s)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return x 

# First iteration of Waste Classifier
class WasteClassifier(nn.Module):
    def __init__(self, get_size=False):
        super(WasteClassifier, self).__init__()
        self.get_size = get_size
        self.resnet_feature_extractor = models.resnet18(pretrained=True)
        self.resnet_feature_extractor = nn.Sequential(*list(self.resnet_feature_extractor.children())[:-3])
        for param in self.resnet_feature_extractor.parameters():
            param.requires_grad = False

        self.conv = nn.Sequential(
            CNN(nin=256, nout=512),
            nn.MaxPool2d(2, 2),
            CNN(nin=512, nout=512),
            nn.Dropout()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        x = self.resnet_feature_extractor(x)
        if self.get_size:
            print("resnet feature layer size: ", x.size())
        x = self.conv(x)
        x = self.flatten(x)
        if self.get_size:
            print("flattened layer size: ", x.size())
        x = self.fc(x)
        x = x.squeeze(1)  # Flatten to [batch_size]
        return x

print(f"Number of parameters in model: {get_num_params(WasteClassifier())}")

def save_model(model, filename=""):
    """ Save a model for inference for later use
    Args:
        model: trained machine learning model to be saved
        filename: name of file
    """
    PATH = ""
    if filename == "":
        PATH = "saved_models/" + str(random.randint(1,100000))
    else:
        PATH = "saved_models/" + filename
    torch.save(model.state_dict(), PATH)

def load_model(model, PATH, device):
    """ Load a previously saved model
    Args:
        model: model with same architecture as the one you're trying to load
        PATH: file path to load from
        device: device (cpu or gpu) to load the model into
    """
    model.load_state_dict(torch.load(PATH, map_location=device))

def get_optimizer(model, config):
    """ Get an optimizer for your model
    Args:
        model: machine learning model
        params: dictionary of hyperparameters for model
    Returns:
        optimizer for model based on params
    """

    if config['optim'] == 'RMS':
        return optim.RMSprop(model.parameters(),
                            lr=config["lr"],
                            momentum=config["mm"],
                            weight_decay=config["wd"])
    elif config['optim'] == 'ADAM':
        return optim.Adam(model.parameters(), 
                            lr=config["lr"],
                            weight_decay=config["wd"])
    else:
        return optim.SGD(model.parameters(), 
                            lr=config["lr"],
                            momentum=config["mm"],
                            weight_decay=config["wd"])
        
@torch.no_grad()
def evaluate(model, val_loader, criterion):
    """ Compute accuracy and loss of the model on the validation set
     Args:
        model: machine learning model
        val_loader: dataloader for the validation set
        criterion: loss function
     Returns:
         acc: the accuracy of the model
         loss: the loss of the model
     """
    total_loss = 0.0
    total_acc = 0.0
    total_examples = 0
    for i, (imgs, labels) in enumerate(val_loader, 0):
        # use GPU if available
        imgs = imgs.to(device)
        labels = labels.to(device)

        # get predictions
        out = model(imgs)
        loss = criterion(out, labels.float())

        # converts to 0 (organic) or 1 (recyclable) and compares with label
        pred = (out >= 0.0).squeeze().long() == labels 

        # calculate accuracy and loss
        total_acc += sum(pred).item()
        total_loss += loss.item()
        total_examples += len(labels)    

    acc = float(total_acc) / total_examples   
    loss = float(total_loss) / (i+1)   
    return acc, loss

@torch.no_grad()
def plot_confusion_matrix(model, dataloader):
    confusion_matrix = np.zeros((2, 2))
    
    for (imgs, labels) in dataloader:
        # use GPU if available
        imgs = imgs.to(device)
        labels = labels.to(device)

        # get predictions
        out = model(imgs)
        pred = (out >= 0.0).squeeze().long()
        for p, l in zip(pred, labels):
            confusion_matrix[p.long(), l.long()] += 1

    # plot heatmap
    plt.figure(figsize=(9,6))
    classes = ['organic', 'recyclable']
    df = pd.DataFrame(confusion_matrix, index=classes, columns=classes).astype(int)
    heatmap = sns.heatmap(df, annot=True, fmt="d")

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0)
    plt.xlabel('Real label')
    plt.ylabel('Predicted label')

    # calculate accuracy
    TP = confusion_matrix[0][0]
    TN = confusion_matrix[1][1]
    acc = (TP + TN) / confusion_matrix.sum()
    print(f"Accuracy: {acc:.8f}")

def plot_trials(dfs):
    """ Plot results of each trial in raytune
        dfs: raytune trial_dataframes
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)  
    for d in dfs.values():
        acc = d.accuracy
        ax.plot(np.arange(1, len(acc)+1), acc, marker='o', label=d['trial_id'][0])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per trial")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def train(config, num_epochs=10, device=None, checkpoint_dir=None):
    """ Train the model and plot the learning curves
     Args:
        config: dictionary of hyperparameters to tune
        device: torch.device object. Either 'cpu' or 'cuda'
     """

    # define model
    model = WasteClassifier()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = get_optimizer(model, config)
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # define dataloader
    train_loader, val_loader = split_data(train_dataset, 
                                    train=True, 
                                    batch_size=config["batch_size"])
   
    train_acc, train_loss = np.zeros(num_epochs), np.zeros(num_epochs)
    val_acc, val_loss = np.zeros(num_epochs), np.zeros(num_epochs)

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_examples = 0

        for i, (imgs, labels) in enumerate(train_loader):
            # use GPU if available
            imgs = imgs.to(device)
            labels = labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # training
            out = model(imgs)
            loss = criterion(out, labels.float())
            loss.backward()
            optimizer.step()

            # converts to 0 (organic) or 1 (recyclable) and compares with label
            pred = (out >= 0.0).squeeze().long() == labels 

            # calculate accuracy and loss
            total_train_acc += sum(pred).item()
            total_train_loss += loss.item()
            total_examples += len(labels)

        train_acc[epoch] = float(total_train_acc) / total_examples   
        train_loss[epoch] = float(total_train_loss) / (i+1)   
        val_acc[epoch], val_loss[epoch] = evaluate(model, val_loader, criterion)
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # report results to raytune
        tune.report(loss=val_loss[epoch], accuracy=val_acc[epoch])

    return

def run_test(num_samples=2, max_num_epochs=2, gpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-4, 1),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "optim": tune.choice(['ADAM', 'SGD']),
        "mm": tune.choice([0.0]),
        "wd": tune.loguniform(1e-5, 5e-3)
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    
    result = tune.run(
        partial(train, num_epochs=max_num_epochs, device=device),
        resources_per_trial={"gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial final validation loss: {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["accuracy"]}')

    best_trained_model = WasteClassifier()
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    dfs = result.trial_dataframes
    return dfs, best_trained_model


if __name__ == "__main__":
    train_dataset = create_dataset(train=True)
    dfs, best_trained_model = run_test(num_samples=10, max_num_epochs=10)

    plot_trials(dfs)

    # test_dataset = create_dataset(train=False)
    # test_loader = split_data(test_dataset,
    #                         train=False,
    #                         batch_size=32)

    train_loader, valid_loader = split_data(train_dataset,
                                            train=True,
                                            train_ratio=0.7,
                                            batch_size=32,
                                            seed=42,
                                            shuffle=True)

    plot_confusion_matrix(best_trained_model, train_loader)
    plot_confusion_matrix(best_trained_model, valid_loader)


    file_name = "best_model"
    save_model(best_trained_model, filename=file_name)
