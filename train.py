from tqdm import tqdm
from functools import partial

tqdm = partial(tqdm, position=0, leave=True)

from torch.cuda.amp import autocast
import torch
import torch.nn.functional as F  # Some of the classes can be directly used as Functions

import copy  # for deep copy I guess
import time
from metrics import Metrics

from utils import *
from tqdm import tqdm


class Trainer:
    """
    Trainer class for training and evaluating a model.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        epochs,
        train_loader,
        valid_loader,
        test_loader,
        model_path,
        wandb,
        classes,
    ):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The model to be trained.
            optimizer: The optimizer for updating model parameters.
            criterion: The loss function.
            device: The device to be used for training.
            epochs (int): The number of epochs to train.
            train_loader: The data loader for the training set.
            valid_loader: The data loader for the validation set.
            test_loader: The data loader for the test set.
            model_path (str): The path to save the best model checkpoint.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model_path = model_path
        self.wandb = wandb
        self.classes = classes

    def save_checkpoint(self):
        """
        Save the model checkpoint.
        """
        torch.save(self.model.state_dict(), self.model_path)

    def cycle(self, plot=False):
        """
        Train and evaluate the model for the specified number of epochs.

        Args:
            plot (bool): Whether to plot the training and validation results.

        Returns:
            test_loss (float): The test loss.
            test_acc (float): The test accuracy.
        """

        best_val_loss = float("inf")
        train_accs = []
        valid_accs = []
        train_losses = []
        valid_losses = []

        self.wandb.watch(self.model, self.criterion, log="all")
        for epoch in tqdm(range(self.epochs)):

            start_time = time.monotonic()

            train_loss, train_acc = self.train(self.train_loader)
            val_loss, val_acc = self.validate(self.valid_loader)
            training_metrics = {
                "train/train_loss": train_loss,
                "train/train_acc": train_acc,
                "val/val_loss": val_loss,
                "val/val_acc": val_acc,
            }
            self.wandb.log(training_metrics)
            self.wandb.log({"val_acc": val_acc})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint()

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            valid_accs.append(val_acc)
            valid_losses.append(val_loss)

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(
                f"""Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s |
                  \tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% |
                  \t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%"""
            )

        test_loss, test_acc = self.test(self.test_loader)
        print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

    def train(self, loader):
        """
        Train the model for one epoch.

        Args:
            loader: The data loader for the training set.

        Returns:
            epoch_loss (float): The average training loss for the epoch.
            epoch_acc (float): The average training accuracy for the epoch.
        """
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0

        for x, y in tqdm(loader, leave = False):
            x = x.to(self.device).float()
            y = y.to(self.device).long()

            self.optimizer.zero_grad()
            # Make predictions
            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(loader), epoch_acc / len(loader)

    def validate(self, loader, get_metrics=False):
        """
        Evaluate the model on the validation set.

        Args:
            loader: The data loader for the validation set.

        Returns:
            epoch_loss (float): The average validation loss for the epoch.
            epoch_acc (float): The average validation accuracy for the epoch.
        """
        epoch_loss = 0
        epoch_acc = 0
        if get_metrics:
            prob_list = []
            label_list = []
            two_d_probs = []

        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(loader, leave = False):
                x = x.to(self.device).float()
                y = y.to(self.device).long()

                y_pred = self.model(x)

                # For Metrics Calculation
                if get_metrics:
                    probabilities = torch.softmax(y_pred, dim=1)
                    prob_list.append(probabilities[:, 1])
                    two_d_probs.append(probabilities)
                    label_list.append(y)

                loss = self.criterion(y_pred, y)
                acc = calculate_accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
        if get_metrics:
            probs = torch.cat(prob_list, dim=0)
            labels = torch.cat(label_list, dim=0)
            two_d_probs = torch.cat(two_d_probs, dim=0)
            metric = Metrics(
                probs, labels, two_d_probs, "Resnet_test", self.wandb, self.classes
            )
            accuracy, auroc, confusion_metrics, f1 = metric.get_stats()
            loss = epoch_loss / len(loader)
            print(f"Accuracy :: {accuracy} | AUC  :: {auroc} | F1 Score :: {f1}")
            return accuracy, auroc, confusion_metrics, f1, loss

        return epoch_loss / len(loader), epoch_acc / len(loader)

    def test(self, loader):
        """
        Evaluate the model on the test set.

        Returns:
            test_loss (float): The test loss.
            test_acc (float): The test accuracy.
        """
        self.model.load_state_dict(torch.load(self.model_path))
        accuracy, auroc, confusion_metrics, f1, loss = self.validate(
            self.test_loader, get_metrics=True
        )
        self.wandb.summary["Accuracy"] = accuracy
        self.wandb.summary["Test_loss"] = loss
        self.wandb.summary["AURoc"] = auroc
        self.wandb.summary["F1_Score"] = f1
        self.wandb.save(self.model_path)
        return accuracy, loss
