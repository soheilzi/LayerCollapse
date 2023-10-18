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
parser.add_argument('--lc1', default=0.2, type=float, help='lc1')
parser.add_argument('--lc2', default=0.2, type=float, help='lc2')
parser.add_argument('--lc3', default=0.1, type=float, help='lc3')
parser.add_argument('--lc4', default=0.1, type=float, help='lc4')
parser.add_argument('--lc5', default=0.01, type=float, help='lc5')
parser.add_argument('--lc6', default=0.01, type=float, help='lc6')
parser.add_argument('--lc7', default=0.01, type=float, help='lc7')
parser.add_argument('--lc8', default=0.01, type=float, help='lc8')
parser.add_argument('--lc9', default=0.01, type=float, help='lc9')
parser.add_argument('--lc10', default=0.01, type=float, help='lc10')
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
parser.add_argument('--fraction', default=0.5, type=float, help='fraction')
parser.add_argument('--patch_size', default=4, type=int, help='patch size')
parser.add_argument('--image_size', default=32, type=int, help='image size')


args = parser.parse_args()
# port vit vanilla imagenet 36237


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Set device
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

# Set dataset
dataloader = get_dataloader(args.dataset, args.batch_size)

# Set model
model = get_model(args.model, args.num_classes, patch_size=args.patch_size, image_size=args.image_size).to(device)
if args.load != "":
    model.load_state_dict(torch.load(args.load))

# Set optimizer
lr = lt.liveVar(args.lr, "lr")
wd = lt.liveVar(args.wd, "wd")
opt = SGD(model.parameters(), lr=lr(), momentum=0.9, weight_decay=wd())

# Register live variables
lc1 = lt.liveVar(args.lc1, "lc1")
lc2 = lt.liveVar(args.lc2, "lc2")
lc3 = lt.liveVar(args.lc3, "lc3")
lc4 = lt.liveVar(args.lc4, "lc4")
lc5 = lt.liveVar(args.lc5, "lc5")
lc6 = lt.liveVar(args.lc6, "lc6")
lc7 = lt.liveVar(args.lc7, "lc7")
lc8 = lt.liveVar(args.lc8, "lc8")
lc9 = lt.liveVar(args.lc9, "lc9")
lc10 = lt.liveVar(args.lc10, "lc10")
fraction = lt.liveVar(args.fraction, "fraction")

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
        if args.reg == "LC" and args.model == "VGG16":
            slope1 = model._modules["fc1"][2].weight
            slope2 = model._modules["fc"][2].weight
            slope3 = model._modules["layer13"][2].weight
            slope4 = model._modules["layer12"][2].weight
            slope5 = model._modules["layer11"][2].weight
            slope6 = model._modules["layer10"][2].weight
            slope7 = model._modules["layer9"][2].weight
            slope8 = model._modules["layer8"][2].weight
            slope9 = model._modules["layer7"][2].weight
            slope10 = model._modules["layer6"][2].weight

            # print with 3 decimal places
            loss = criterion(outputs, labels) + lc1() * (1. - slope1) ** 2 + lc2() * (1 - slope2) ** 2 + lc3() * (1 - slope3) ** 2 + lc4() * (1 - slope4) ** 2 + lc5() * (1 - slope5) ** 2 + lc6() * (1 - slope6) ** 2 + lc7() * (1 - slope7) ** 2 + lc8() * (1 - slope8) ** 2 + lc9() * (1 - slope9) ** 2 + lc10() * (1 - slope10) ** 2
            print(f"slope1: {slope1.item():.3f}, slope2: {slope2.item():.3f}, slope3: {slope3.item():.3f}, slope4: {slope4.item():.3f}, slope5: {slope5.item():.3f}, slope6: {slope6.item():.3f}, slope7: {slope7.item():.3f}, slope8: {slope8.item():.3f}, slope9: {slope9.item():.3f}, slope10: {slope10.item():.3f}")
            # print((lc1() * (1. - slope1) ** 2 + lc2() * (1 - slope2) ** 2 + lc3() * (1 - slope3) ** 2 + lc4() * (1 - slope4) ** 2 + lc5() * (1 - slope5) ** 2 + lc6() * (1 - slope6) ** 2 + lc7() * (1 - slope7) ** 2 + lc8() * (1 - slope8) ** 2 + lc9() * (1 - slope9) ** 2 + lc10() * (1 - slope10) ** 2).item())
            # print(lc1(), lc2(), lc3(), lc4(), lc5(), lc6(), lc7(), lc8(), lc9(), lc10())
            # print(slope1.requires_grad)

        elif args.reg == "LC" and args.model == "mixer":
            loss = criterion(outputs, labels) + get_model_linear_loss(model, fraction=fraction()) * lc1()
        elif args.reg == "LC" and args.model == "timm_vit":
            loss = criterion(outputs, labels) + get_model_linear_loss(model, fraction=fraction()) * lc1()
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