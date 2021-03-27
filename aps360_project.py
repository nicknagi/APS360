import time
import random
import multiprocessing

import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, models, transforms

import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from shutil import copyfile

import pandas as pd
from pandas import DataFrame
import seaborn as sns

torch.multiprocessing.set_sharing_strategy('file_system')

"""## Setup GPU"""

use_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
if device.type == 'cuda':
    print('CUDA is available. Training on GPU')
else:
    print('CUDA is not available. Training on CPU')

train_data_dir = "aps360_project_db/DATASET/TRAIN"
test_data_dir = "aps360_project_db/DATASET/TEST"


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
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

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


def create_dataset(training_data=True):
    """ Load data from file system and create a dataset
    Args:
        training_data: choose between training set and test set
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
    if training_data:
        return datasets.ImageFolder(train_data_dir, transform=data_transform)
    else:
        return datasets.ImageFolder(test_data_dir, transform=data_transform)


def split_data(dataset, is_train=True, train_ratio=0.7, batch_size=25,
               seed=1000, shuffle=True):
    """ Load data from dataset and split it into train/valid/test sets
    Args:
        dataset: dataset containing all data
        is_train: determines if dataset is for the train or the test set. 
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
    prefetch_factor = 4
    pin_mem = False
    if torch.cuda.is_available():
        pin_mem = True

    if is_train:
        # calculate indices to split data at
        split = int(np.floor(train_ratio * num_data))

        # train/valid split
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = torch.utils.data.Subset(dataset, train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # create Data loaders
        training_dataloader = DataLoader(dataset,
                                         batch_size=batch_size,
                                         sampler=ImbalancedDatasetSampler(train_sampler),
                                         num_workers=num_workers,
                                         pin_memory=pin_mem,
                                         prefetch_factor=prefetch_factor)

        valid_dataloader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=valid_sampler,
                                      num_workers=num_workers,
                                      pin_memory=pin_mem,
                                      prefetch_factor=prefetch_factor)

        return training_dataloader, valid_dataloader

    else:
        # all data placed in test set
        test_idx = indices
        test_sampler = SubsetRandomSampler(test_idx)

        test_dataloader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=test_sampler,
                                     num_workers=num_workers,
                                     pin_memory=pin_mem,
                                     prefetch_factor=prefetch_factor)

        return test_dataloader


def split_classes(dataloader, seed=1000, batch_size=25):
    """ Splits data between 2 classes (organic and recyclable)
    Args:
        :param dataloader: dataloader of data to split
        :param seed: random seed. Use same seed as when you split data (preferably)
        :param batch_size: batch size duh

    Returns:
        organic_dataset: organic data dataset
        recycle_dataset: recyclable data dataset
        organic_dataloader: organic data dataloader
        recycle_dataloader: recyclable data dataloader
    """
    # set seed
    torch.manual_seed(seed)

    # dataloader parameters
    num_workers = 0
    pin_mem = False
    if torch.cuda.is_available():
        pin_mem = True

    organic_imgs = []
    organic_labels = []
    recyclable_imgs = []
    recyclable_labels = []

    # loop through data in dataloader and split into 2 classes
    # organic == 0, recyclable == 1
    for img_batch, label_batch in dataloader:
        for j in range(len(label_batch)):
            if label_batch[j] == 0:
                organic_imgs.append(img_batch[j])
                organic_labels.append(label_batch[j])
            else:
                recyclable_imgs.append(img_batch[j])
                recyclable_labels.append(label_batch[j])

    # convert to tensors
    t_organic_imgs = torch.stack(organic_imgs)
    t_organic_labels = torch.stack(organic_labels)
    t_recyclable_imgs = torch.stack(recyclable_imgs)
    t_recyclable_labels = torch.stack(recyclable_labels)

    # convert to Dataset objects
    organic_dataset = TensorDataset(t_organic_imgs, t_organic_labels)
    recycle_dataset = TensorDataset(t_recyclable_imgs, t_recyclable_labels)

    # construct DataLoaders
    organic_dataloader = DataLoader(organic_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_mem)

    recycle_dataloader = DataLoader(recycle_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_mem)

    return organic_dataset, recycle_dataset, organic_dataloader, recycle_dataloader


"""# 4. Define Model"""


def get_num_params(net):
    """determine the size of the model
    Args:
        net: machine learning model
    Returns:
        size of model
    """
    return sum(p.numel() for p in net.parameters())


# Base CNN model
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
        self.conv = nn.Sequential(
            CNN(nin=3, nout=10),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            CNN(nin=10, nout=30),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            CNN(nin=30, nout=64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(36864, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        if self.get_size:
            print("flattened layer size: ", x.size())
        x = self.fc(x)
        x = x.squeeze(1)  # Flatten to [batch_size]
        return x


"""# 5. Training Code"""


def save_model(model, filename=""):
    """ Save a model for inference for later use
    Args:
        model: trained machine learning model to be saved
        filename: name of file
    """
    if filename == "":
        path = "saved_models/" + str(random.randint(1, 100000))
    else:
        path = "saved_models/" + filename
    torch.save(model.state_dict(), path)


def load_model(model, PATH, device):
    """ Load a previously saved model
    Args:
        model: model with same architecture as the one you're trying to load
        PATH: file path to load from
        device: device (cpu or gpu) to load the model into
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


def get_optimizer(_model, _params):
    """ Get an optimizer for your model
    Args:
        _model: machine learning model
        _params: dictionary of hyperparameters for model
    Returns:
        optimizer for model based on _params
    """

    if _params.get('optim') == 'RMS':
        return optim.RMSprop(_model.parameters(),
                             lr=_params.get('lr', 0.01),
                             momentum=_params.get('mm', 0.0),
                             weight_decay=_params.get('wd', 0.0))
    elif _params.get('optim') == 'ADAM':
        return optim.Adam(_model.parameters(),
                          lr=_params.get('lr', 0.001),
                          weight_decay=_params.get('wd', 0.0))
    else:
        return optim.SGD(_model.parameters(),
                         lr=_params.get('lr', 0.01),
                         momentum=_params.get('mm', 0.0),
                         weight_decay=_params.get('wd', 0.0))


@torch.no_grad()
def evaluate(_model, val_loader, criterion):
    """ Compute accuracy and loss of the model on the validation set
     Args:
        _model: machine learning model
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
        out = _model(imgs).squeeze()
        loss = criterion(out, labels.float())

        # converts to 0 (organic) or 1 (recyclable) and compares with label
        pred = (out >= 0.0).squeeze().long() == labels

        # calculate accuracy and loss
        total_acc += sum(pred).item()
        total_loss += loss.item()
        total_examples += len(labels)

    acc = float(total_acc) / total_examples
    loss = float(total_loss) / (i + 1)
    return acc, loss


@torch.no_grad()
def test_accuracy(model_to_evaluate, dataloader):
    """ Compute accuracy of the model on the test set
     Args:
        model_to_evaluate: machine learning model
        dataloader: dataloader for the test set
     Returns:
         acc: the accuracy of the model
     """
    total_acc = 0.0
    total_examples = 0
    for imgs, labels in dataloader:
        # use GPU if available
        imgs = imgs.to(device)
        labels = labels.to(device)

        # get predictions
        out = model_to_evaluate(imgs)

        # converts to 0 (organic) or 1 (recyclable) and compares with label
        pred = (out >= 0.0).squeeze().long() == labels

        # calculate accuracy
        total_acc += sum(pred).item()
        total_examples += len(labels)

    acc = float(total_acc) / total_examples
    return acc


@torch.no_grad()
def plot_confusion_matrix(_model, dataloader):
    num_class = 2
    confusion_matrix = np.zeros((num_class, num_class))

    for (imgs, labels) in dataloader:
        # use GPU if available
        imgs = imgs.to(device)
        labels = labels.to(device)

        # get predictions
        out = _model(imgs).squeeze()
        pred = (out >= 0.0).squeeze().long()
        for p, l in zip(pred, labels):
            confusion_matrix[p.long(), l.long()] += 1

    plt.figure(figsize=(9, 6))
    classes = ['organic', 'recyclable']
    df = pd.DataFrame(confusion_matrix, index=classes, columns=classes).astype(int)
    heatmap = sns.heatmap(df, annot=True, fmt="d")

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0)
    plt.xlabel('Real label')
    plt.ylabel('Predicted label')

    plt.show()


def plot_loss_acc(num_epochs, train_loss, train_acc, val_loss, val_acc):
    """ Plot training curves
    Args:
        num_epochs: number of epochs to plot
        train_loss: list of training losses
        train_acc: list of training accuracies
        val_loss: list of validation losses
        val_acc: list of validation accuracies
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    fig.subplots_adjust(wspace=0.4)

    axes[0].set_title("Training & Validation Loss")
    axes[0].plot(range(1, num_epochs + 1), train_loss, label="Training")
    axes[0].plot(range(1, num_epochs + 1), val_loss, label="Validation")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc='best')

    axes[1].set_title("Training & Validation Accuracy")
    axes[1].plot(range(1, num_epochs + 1), train_acc, label="Training")
    axes[1].plot(range(1, num_epochs + 1), val_acc, label="Validation")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc='best')

    plt.show()


def train(model_to_train, training_loader, val_loader, parameters, plot=False, show_updates=False):
    """ Train the model and plot the learning curves
     Args:
        model_to_train: machine learning model
        training_loader: dataloader for the training set
        val_loader: dataloader for the validation set
        parameters: dictionary of hyperparameters for model
        plot: flag to display training curves
        show_updates: flag to display training updates each epoch
     """

    num_epochs = parameters.get('epoch', 10)
    criterion = nn.BCEWithLogitsLoss()  # expects targets to be of type torch.long
    optimizer = get_optimizer(model_to_train, parameters)

    train_acc, train_loss = np.zeros(num_epochs), np.zeros(num_epochs)
    val_acc, val_loss = np.zeros(num_epochs), np.zeros(num_epochs)

    n = 0  # the number of iterations
    start_time = time.time()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_examples = 0

        for i, (imgs, labels) in enumerate(training_loader):
            # use GPU if available
            imgs = imgs.to(device)
            labels = labels.to(device)

            # training
            out = model_to_train(imgs).squeeze()
            loss = criterion(out, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # converts to 0 (organic) or 1 (recyclable) and compares with label
            pred = (out >= 0.0).squeeze().long() == labels

            # calculate accuracy and loss
            total_train_acc += sum(pred).item()
            total_train_loss += loss.item()
            total_examples += len(labels)

        train_acc[epoch] = float(total_train_acc) / total_examples
        train_loss[epoch] = float(total_train_loss) / (i + 1)
        val_acc[epoch], val_loss[epoch] = evaluate(model_to_train, val_loader, criterion)

        # print updates each epoch
        if show_updates:
            print(f"Epoch {epoch + 1: <2} | "
                  f"Train acc: {train_acc[epoch]:<12.8f} | "
                  f"Train loss: {train_loss[epoch]:<12.8f} | "
                  f"Val acc: {val_acc[epoch]:<12.8f} | "
                  f"Val loss: {val_loss[epoch]:<12.8f}")

    # print final updates
    if not show_updates:
        print(f"Epoch {epoch + 1: <2} | "
              f"Train acc: {train_acc[epoch]:<12.8f} | "
              f"Train loss: {train_loss[epoch]:<12.8f} | "
              f"Val acc: {val_acc[epoch]:<12.8f} | "
              f"Val loss: {val_loss[epoch]:<12.8f}")

    # time taken 
    end_time = time.time()
    print(f"Total time: {end_time - start_time: .2f}s")

    if plot:
        plot_loss_acc(num_epochs, train_loss, train_acc, val_loss, val_acc)

    return


# -----------------------------------------------------------------------------------------------------
params = {'batch': 64, 'epoch': 5, 'optim': 'ADAM', 'lr': 0.000285145, 'mm': 0.0, 'wd': 0, 'seed': 42}

# create train dataset
train_dataset = create_dataset(training_data=True)
test_dataset = create_dataset(training_data=False)

test_loader = split_data(test_dataset,
                         is_train=False,
                         batch_size=params.get('batch'),
                         seed=params.get('seed'),
                         shuffle=True)

train_loader, valid_loader = split_data(train_dataset,
                                        is_train=True,
                                        train_ratio=0.7,
                                        batch_size=params.get('batch'),
                                        seed=params.get('seed'),
                                        shuffle=True)
#
model = WasteClassifier(get_size=False)
model.to(device)

# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model.fc = nn.Linear(num_ftrs, 1)
# model.to(device)

print(f"Number of parameters in model: {get_num_params(model)}")
train(model, train_loader, valid_loader, params, plot=True, show_updates=True)

print(f"Test Accuracy: {test_accuracy(model, test_loader):.8f}")

"""
Save a model to local file system
"""
# save_model(model)

"""
Load a saved model from local file system
"""
# new_model = WasteClassifier()
# load_model(new_model, "/content/saved_models/58870", device)
# new_model.to(device)

# Test accuracy for false positives and negatives
neg_ds, pos_ds, neg_dl, pos_dl = split_classes(test_loader, params["seed"], params["batch"])
print(f"Accuracy on Organic set: {test_accuracy(model, neg_dl)},\
     Accuracy on Recyclable set: {test_accuracy(model, pos_dl)}")

# plot_confusion_matrix(model, neg_dl)
# plot_confusion_matrix(model, pos_dl)
