import copy
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import Union, List
from utils import *

import numpy as np
import torch
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from tqdm.auto import tqdm

assert torch.cuda.is_available(), \
"CUDA support is not available."

import pickle

import LiveTune as lt

import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument("--wd", default=0.0, type=float, help="weight decay")
parser.add_argument('--lc1', default=0.05, type=float, help='lc1')
parser.add_argument('--lc2', default=0.2, type=float, help='lc2')
parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--device', default=0, type=int, help='device')
parser.add_argument('--model', default="VGG16", type=str, help='model')
parser.add_argument('--dataset', default="cifar10", type=str, help='dataset')
parser.add_argument('--save_dir', default="./data/", type=str, help='save directory')
parser.add_argument('--save_name', default="test", type=str, help='save name')
parser.add_argument('--reg', default="none", type=str, help='regularization')
parser.add_argument('--reg_strength', default=0.0, type=float, help='regularization strength')
parser.add_argument('--load', default="", type=str, help='load model')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--use_scheduler', action="store_true", help='use scheduler')

args = parser.parse_args()
# port wd 51855
# port vannila small 
# port LC 42343
# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Set device
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

# Set dataset
dataloader = get_dataloader(args.dataset, args.batch_size)

# Set model
model = get_model(args.model, args.num_classes).to(device)
if args.load != "":
    model.load_state_dict(torch.load(args.load))

# Set optimizer
lr = lt.liveVar(args.lr, "lr")
wd = lt.liveVar(args.wd, "wd")
opt = SGD(model.parameters(), lr=lr(), momentum=0.9, weight_decay=wd())

# Register live variables
lc1 = lt.liveVar(args.lc1, "lc1")
lc2 = lt.liveVar(args.lc2, "lc2")

old_lts = {"lr": lr(), "wd": wd(), "lc1": lc1(), "lc2": lc2()}
# Set loss function
criterion = nn.CrossEntropyLoss()

# Set save trigger
save_trigger = lt.liveTrigger("save")

train_losses = []
train_accs = []
val_losses = []
val_accs = []

# Train
for epoch in tqdm(range(args.epochs)):
    # Train
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for i, (inputs, labels) in enumerate(dataloader["train"]):
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = model(inputs)

        # Regularized loss
        if args.reg == "LC":
            slope1 = model._modules["fc"][2].weight
            slope2 = model._modules["fc1"][2].weight
            print(slope1)
            print(slope2)
            loss = criterion(outputs, labels) + lc1() * (1 - slope1) ** 2 + lc2() * (1 - slope2) ** 2
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        train_acc += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(dataloader["train"])
    train_acc /= len(dataloader["train"].dataset)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    val_losses.append(eval(model, dataloader["val"], criterion, device))
    val_accs.append(evaluate(model, dataloader["val"], device=device))

    # Save
    if save_trigger():
        torch.save(model.state_dict(), args.save_dir + args.save_name + ".pth")
        with open(args.save_dir + args.save_name + ".pkl", "wb") as f:
            pickle.dump({
                "train_losses": train_losses,
                "train_accs": train_accs,
                "val_losses": val_losses,
                "val_accs": val_accs,
            }, f)


    # Print
    print("Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}".format(epoch, train_losses[-1], train_accs[-1], val_losses[-1], val_accs[-1]))

    # Scheduler
    if args.use_scheduler and epoch + 1 % 20 == 0:
        with lr.lock:
            lr.var_value = lr.var_value * 0.5 ** (epoch // 20)
    # LiveTune
    if old_lts["lr"] != lr() or old_lts["wd"] != wd():
        old_lts["lr"] = lr()
        old_lts["wd"] = wd()
        opt = SGD(model.parameters(), lr=lr(), momentum=0.9, weight_decay=wd())
        

torch.save(model.state_dict(), args.save_dir + args.save_name + ".pth")
with open(args.save_dir + args.save_name + ".pkl", "wb") as f:
    pickle.dump({
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs
    }, f)