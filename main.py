import torch
import torch.nn as nn
import torch.nn.functional as F  # Some of the classes can be directly used as Functions
import torch.utils.data as data

import wandb
from train import Trainer
from utils import *
from model import *
from data import *

import sys

models = {

    'resnet101': 'resnet101.pth',
    'resnet50' : 'resnet50.pth',
    'resnext50d_32x4d': 'resnext.pth',
    
}

FuncMap = {
    'resnet101' : ResNetModel,
    'resnet50' : ResNetModel,
    'resnext50d_32x4d' : ResNextModel,
    
}

model_name = sys.argv[1]
model_path = models[model_name]
model_func = FuncMap[model_name]


# Freezing the Seed for Reproducibility
seed_everything(seed=42)  # The Answer to Everything



# Creating Dataset
ds = ProductsDataset(root_dir="PnG_HC_clf_torch")
train_data = ds("train", transforms=image_transforms["train"])
valid_data = ds("valid", transforms=image_transforms["valid"])
test_data  = ds("test", transforms=image_transforms["valid"])

# Dataset Classes
classes = train_data.classes

# Logging In to wandb
wandb.login()

# 🐝 initialise a wandb run
wandb.init(
    project="PandG_HC",
    config={
        "LR": 1e-3,
        "NUM_EPOCHS": 30,
        "BATCH_SIZE": 16,
        "MODEL_PTH": model_path,
        "NUM_CLASSES" : len(classes)
    },
)
config = wandb.config

# Creating DataLoader
dataloaders = {
    "train": data.DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    ),
    "valid": data.DataLoader(
        valid_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    ),
    "test": data.DataLoader(
        test_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    ),
}


# Model and Setups
# model = ResNextModel(num_classes=2)
model = model_func(num_classes=config.NUM_CLASSES, model_name=model_name, pretrained= True)

print(f"The model has {count_parameters(model):,} trainable parameters")

# Defining Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
criterion = nn.CrossEntropyLoss()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initalizing Model
model = model.to(DEVICE)
criterion = criterion.to(DEVICE)

# Instantiating Trainer
trainer = Trainer(
    model,
    optimizer,
    criterion,
    DEVICE,
    config.NUM_EPOCHS,
    dataloaders["train"],
    dataloaders["valid"],
    dataloaders["test"],
    config.MODEL_PTH,
    wandb,
    classes,
)

# Running Training cycle
trainer.cycle()

# Completing wandb logging
wandb.finish()
