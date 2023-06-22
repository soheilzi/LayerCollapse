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
from matplotlib import pyplot as plt
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
import torchvision.models as models

from torchprofile import profile_macs
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn.parallel import DataParallel

import torch.multiprocessing as mp

import argparse

assert torch.cuda.is_available(), "CUDA support is not available."

# get command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_hinging_steps", type=int, default=10)
parser.add_argument("--max_finetune_epochs", type=int, default=10)
parser.add_argument("--acc_loss_threshold", type=float, default=2)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--nprocs", type=int, default=3)
parser.add_argument("--log_dir", type=str, default="../../../../../../../data/soheil/layercollapse/imagenet1")
args = parser.parse_args()

# rank = dist.get_rank()


# create model and move it to GPU with id rank

# define model
model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')

if __name__=="__main__":
    # if mp.current_process().name == "MainProcess":
    #     tqdm.set_lock(mp.RLock())
    # else:
    #     tqdm.set_lock(None)

    # remove_relu_imagenet_vgg16(model, 4, args)
    model.classifier._modules[str(4)] = nn.LeakyReLU(negative_slope=nn.Parameter(torch.tensor(1), requires_grad=False), inplace=True)
    model.load_state_dict(torch.load('temp_best_model.pth'))    
    collapse_layers_imagenet_vgg16(model, 4)
    remove_relu_imagenet_vgg16(model, 1, args)
    collapse_layers_imagenet_vgg16(model, 1)