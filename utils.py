import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn


def seed_everything(seed):
    """
    Set random seed for reproducibility across different libraries and frameworks.

    Args:
        seed (int): Seed value to set for random number generators.
    """
    # Set random seed for Python's built-in random module
    random.seed(seed)

    # Set random seed for os.environ to control hashing of environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set random seed for numpy
    np.random.seed(seed)

    # Set random seed for PyTorch on CPU
    torch.manual_seed(seed)

    # Set random seed for PyTorch on GPU (if available)
    torch.cuda.manual_seed(seed)

    # Enable deterministic behavior of cudnn (CUDA backend for PyTorch)
    torch.backends.cudnn.deterministic = True

    # Enable benchmark mode in cudnn for improved performance
    torch.backends.cudnn.benchmark = True


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad == True])


def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(m.bias.data, 0)


def plot_results(train_acc, valid_acc, train_loss, valid_loss, nb_epochs):
    """
    Plot the training and validation results (accuracy and loss) over epochs.

    Args:
        train_acc (list): List of training accuracies at each epoch.
        valid_acc (list): List of validation accuracies at each epoch.
        train_loss (list): List of training losses at each epoch.
        valid_loss (list): List of validation losses at each epoch.
        nb_epochs (int): Total number of epochs.
    """
    epochs = [i for i in range(nb_epochs)]

    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)

    # Plot training and validation accuracy
    ax[0].plot(epochs, train_acc, "go-", label="Training Accuracy")
    ax[0].plot(epochs, valid_acc, "ro-", label="Validation Accuracy")
    ax[0].set_title("Training & Validation Accuracy")
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    # Plot training and validation loss
    ax[1].plot(epochs, train_loss, "go-", label="Training Loss")
    ax[1].plot(epochs, valid_loss, "ro-", label="Validation Loss")
    ax[1].set_title("Training & Validation Loss")
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")

    # Display the plot
    plt.show()


def calculate_accuracy(y_pred, y):
    # First we will get the index with maximum confidence
    top_pred = y_pred.argmax(
        1, keepdim=True
    )  # Taking Max at Second DimensionKeeping as Batch and Max Value
    # Now we wil reshape  Y to be the same as preds as a check
    y = y.view_as(top_pred)
    # Now we will equate the Y and and Y_pred
    correct = top_pred.eq(y).sum()  # Equating the True and Taking the Sum of all Trues
    # Finally Accuracy is True Divide by all
    acc = correct.float() / y.shape[0]
    return acc


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


"""
early_stopper = EarlyStopper(patience=3, min_delta=10)
for epoch in np.arange(n_epochs):
    train_loss = train_one_epoch(model, train_loader)
    validation_loss = validate_one_epoch(model, validation_loader)
    if early_stopper.early_stop(validation_loss):             
        break

"""


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
