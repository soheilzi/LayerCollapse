import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
import copy

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

from utils import get_dataloader

assert torch.cuda.is_available(), "CUDA support is not available."

# get command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_hinging_steps", type=int, default=10)
parser.add_argument("--max_finetune_epochs", type=int, default=10)
parser.add_argument("--acc_loss_threshold", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--nprocs", type=int, default=3)
parser.add_argument("--log_dir", type=str, default="../../../../../../../data/soheil/layercollapse/imagenet1")
args = parser.parse_args()

model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



# dataloader = get_dataloader("imagenet", batch_size=256)

num_finetune_epochs = 10

@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataloader: DataLoader, 
  verbose=False,
  device=None,
) -> float:
  # model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, 
                                disable=not verbose):
        # Move the data from CPU to GPU
        if device is None:
            inputs = inputs.cuda()
            targets = targets.cuda()
        else:
            inputs = inputs.to(non_blocking=True, device=device)
            targets = targets.to(non_blocking=True, device=device)

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()

def train_multiprocess(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  scheduler: LambdaLR,
  callbacks = None,
  device_id = 0
) -> None:
    model.train()
    # Create a distributed sampler
    # sampler = torch.utils.data.distributed.DistributedSampler(dataloader)
    
    # # # Create a new dataloader with the distributed sampler
    # dataloader = DataLoader(dataloader.dataset, sampler=sampler, batch_size=dataloader.batch_size)

    for inputs, targets in tqdm(dataloader, desc='train', leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.to(non_blocking=True, device=device_id)
        targets = targets.to(non_blocking=True, device=device_id)
        # print(f'    {device_id} {inputs.device} {targets.device}')
        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()

        # Forward inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward propagation
        loss.backward()

        # Update optimizer and LR scheduler
        optimizer.step()
        scheduler.step()

        if callbacks is not None:
            for callback in callbacks:
                callback()

def finetune(rank, world_size, model, lr, num_finetune_epochs):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # print cuda is available and device is correct
    print(f'    {rank} {torch.cuda.is_available()} {torch.cuda.current_device()} {torch.cuda.get_device_name()}')

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
    criterion = nn.CrossEntropyLoss()

    best_sparse_model_checkpoint = dict()
    best_accuracy = 0
    test_accuracy = []
    train_accuracy = []
    print(f'Finetuning changed Model')

    dataloader = get_dataloader("imagenet", batch_size=256, num_workers=10)

    sampler = torch.utils.data.distributed.DistributedSampler(dataloader['train'].dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataloader['val'].dataset, num_replicas=world_size, rank=rank)


    dataloader['train'] = DataLoader(dataloader['train'].dataset, sampler=sampler, batch_size=dataloader['train'].batch_size, pin_memory=True, num_workers=10)
    dataloader['val'] = DataLoader(dataloader['val'].dataset, batch_size=dataloader['val'].batch_size, sampler=val_sampler, pin_memory=True, num_workers=10)


    for epoch in range(num_finetune_epochs):
        # At the end of each train iteration, we have to apply the pruning mask 
        #    to keep the model sparse during the training
        # print(f"here at epoch {epoch}")
        train_multiprocess(model, dataloader['train'], criterion, optimizer, scheduler,
            callbacks=None, device_id=device)
        accuracy = evaluate(model, dataloader['val'], device=device)
        # if acc_loss_threshold is not None and accuracy > acc_loss_threshold + initial_acc:
        #     print(f'    Early break on epoch {epoch+1} with accuracy {accuracy:.2f}%')
        #     break
        test_accuracy.append(accuracy)
        train_accuracy.append(evaluate(model, dataloader['train'], device=device))
        is_best = accuracy > best_accuracy
        if is_best:
            best_sparse_model_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_accuracy = accuracy
        print(f'    Epoch {epoch+1} proc {rank} test Accuracy {accuracy:.2f}% / Best test Accuracy: {best_accuracy:.2f}%')
        print(f'    Epoch {epoch+1} proc {rank} train Accuracy {train_accuracy[-1]:.2f}% / Best train Accuracy: {max(train_accuracy):.2f}%')
    cleanup()
    return train_accuracy, test_accuracy



def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size, model, args.lr, num_finetune_epochs),
             nprocs=world_size,
             join=True)
    

if __name__=="__main__":
    if mp.current_process().name == "MainProcess":
        tqdm.set_lock(mp.RLock())
    else:
        tqdm.set_lock(None)
    run_demo(finetune, args.nprocs)