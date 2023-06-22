import copy
import time
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from torchvision.datasets import *
from torchvision.transforms import *

from torchprofile import profile_macs
import numpy as np

# # import Subset function of torchvision
# from torchvision.datasets.utils import 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.multiprocessing as mp


# old version
def train(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  scheduler: LambdaLR,
  callbacks = None
) -> None:
    model.train()
    # Create a distributed sampler
    # sampler = torch.utils.data.distributed.DistributedSampler(dataloader)
    
    # # Create a new dataloader with the distributed sampler
    # dataloader = DataLoader(dataloader.dataset, sampler=sampler, batch_size=dataloader.batch_size)

    for inputs, targets in tqdm(dataloader, desc='train', leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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



@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataloader: DataLoader, 
  verbose=True,
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
            inputs = inputs.to(device)
            targets = targets.to(device)

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()

def finetune2(rank, world_size, model, lr, num_finetune_epochs, initial_acc, args):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # print cuda is available and device is correct
    # print(f'    {rank} {torch.cuda.is_available()} {torch.cuda.current_device()} {torch.cuda.get_device_name()}')

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

        # if args.acc_loss_threshold is not None and accuracy + args.acc_loss_threshold > initial_acc:
        #     print(f'    Early break on epoch {epoch+1} with accuracy {accuracy:.2f}%')
        #     torch.save(model.module.state_dict(), 'temp_best_model.pth')
        #     break

        test_accuracy.append(accuracy)
        train_accuracy.append(evaluate(model, dataloader['train'], device=device))
        is_best = accuracy > best_accuracy
        if is_best:
            best_sparse_model_checkpoint['state_dict'] = copy.deepcopy(model.module.state_dict())
            best_accuracy = accuracy
            if rank == 0:
                torch.save(best_sparse_model_checkpoint['state_dict'], 'temp_best_model.pth')
        print(f'    Epoch {epoch+1} proc {rank} test Accuracy {accuracy:.2f}% / Best test Accuracy: {best_accuracy:.2f}%')
        print(f'    Epoch {epoch+1} proc {rank} train Accuracy {train_accuracy[-1]:.2f}% / Best train Accuracy: {max(train_accuracy):.2f}%')
    cleanup()
    return train_accuracy, test_accuracy

def run_finetune(world_size, model, lr, num_finetune_epochs, initial_acc, args):
    mp.spawn(finetune2,
             args=(world_size, model, lr, num_finetune_epochs, initial_acc, args),
             nprocs=world_size,
             join=True)
    # try:
    #     cleanup()
    # except:
    #     print("cleanup not required")
    #     pass


def finetune(model, dataloader, num_finetune_epochs=5, lr=0.01, multi_gpu=False, acc_loss_threshold=None, initial_acc=0, device_id=0):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
    criterion = nn.CrossEntropyLoss()

    best_sparse_model_checkpoint = dict()
    best_accuracy = 0
    test_accuracy = []
    train_accuracy = []
    print(f'Finetuning changed Model')
    for epoch in range(num_finetune_epochs):
        # At the end of each train iteration, we have to apply the pruning mask 
        #    to keep the model sparse during the training
        # print(f"here at epoch {epoch}")
        if multi_gpu:
            train_multiprocess(model, dataloader['train'], criterion, optimizer, scheduler,
                callbacks=None, device_id=device_id)
        else:
            train(model, dataloader['train'], criterion, optimizer, scheduler,
                callbacks=None)
        accuracy = evaluate(model, dataloader['val'])
        if acc_loss_threshold is not None and accuracy > acc_loss_threshold + initial_acc:
            print(f'    Early break on epoch {epoch+1} with accuracy {accuracy:.2f}%')
            break
        test_accuracy.append(accuracy)
        train_accuracy.append(evaluate(model, dataloader['train'], verbose=False))
        is_best = accuracy > best_accuracy
        if is_best:
            best_sparse_model_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_accuracy = accuracy
        print(f'    Epoch {epoch+1} test Accuracy {accuracy:.2f}% / Best test Accuracy: {best_accuracy:.2f}%')
        print(f'    Epoch {epoch+1} train Accuracy {train_accuracy[-1]:.2f}% / Best train Accuracy: {max(train_accuracy):.2f}%')
    return train_accuracy, test_accuracy

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def get_dataloader(dataset_name="cifar100", batch_size=512, num_workers=4):
    image_size = 224
    transforms = {
        "train": Compose([
            Resize(256),
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "val": Compose([
            Resize(256),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
    dataset = {}
    for split in ["train", "val"]:
       # get cifar100 dataset
        if dataset_name == "cifar100":
            dataset[split] = CIFAR100(
                root="data/cifar100",
                train=(split == "train"),
                download=True,
                transform=transforms[split],
            )
        elif dataset_name == "cifar10":
            dataset[split] = CIFAR10(
                root="data/cifar10",
                train=(split == "train"),
                download=True,
                transform=transforms[split],
            )
        elif dataset_name == "imagenet":
            dataset[split] = ImageNet(
                root="/data/soheil/datasets/imagenet",
                split=split,
                transform=transforms[split],
            )
    dataloader = {}
    for split in ['train', 'val']:
        dataloader[split] = DataLoader(
            torch.utils.data.Subset(dataset[split], list(range(0, len(dataset[split]), 10 if split == 'train' else 10))),
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloader

def evaluate_model_imagenet(model, dataloader, count_nonzero_only=False):
    model_test_accuracy = evaluate(model, dataloader['val'])
    print(f"model has test accuracy={model_test_accuracy:.2f}%")
    model_train_accuracy = evaluate(model, dataloader['train'])
    model_size = get_model_size(model, count_nonzero_only=count_nonzero_only)
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    model_macs = get_model_macs(model, dummy_input)
    # print(f"model has test accuracy={model_test_accuracy:.2f}%")
    print(f"model has train accuracy={model_train_accuracy:.2f}%")
    print(f"model has size={model_size/MiB:.2f} MiB")
    print(f"model has macs={model_macs/1e9:.2f} Gmacs")
    model.eval()
    average_time = 0
    with torch.no_grad():
        for i in range(1000):
            start = time.time()
            output = model(dummy_input)
            end = time.time()
            average_time += (end - start)

    print(f"average inference time is {average_time/1000:.4f} seconds")
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"model has {get_num_parameters(model, count_nonzero_only)/1e6:.2f} M parameters")

def evaluate_model(model, dataloader, count_nonzero_only=False):
    model_test_accuracy = evaluate(model, dataloader['val'])
    model_train_accuracy = evaluate(model, dataloader['train'])
    model_size = get_model_size(model, count_nonzero_only=count_nonzero_only)
    dummy_input = torch.randn(1, 3, 32, 32).cuda()
    model_macs = get_model_macs(model, dummy_input)
    print(f"model has test accuracy={model_test_accuracy:.2f}%")
    print(f"model has train accuracy={model_train_accuracy:.2f}%")
    print(f"model has size={model_size/MiB:.2f} MiB")
    print(f"model has macs={model_macs/1e9:.2f} Gmacs")
    model.eval()
    average_time = 0
    with torch.no_grad():
        for i in range(1000):
            start = time.time()
            output = model(dummy_input)
            end = time.time()
            average_time += (end - start)

    print(f"average inference time is {average_time/1000:.4f} seconds")
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"model has {get_num_parameters(model, count_nonzero_only)/1e6:.2f} M parameters")


def change_module(model, old_module, _number, new_module):
    model.backbone.classifier._modules[_number] = new_module

def remove_relu(model, dataloader, old_module, _number, finetune_epochs, lr):
    change_module(model, old_module, _number, nn.LeakyReLU(negative_slope=nn.Parameter(torch.tensor(0.1), requires_grad=False), inplace=True))
    finetune(model, dataloader, finetune_epochs, lr)
    change_module(model, nn.LeakyReLU, _number, nn.LeakyReLU(negative_slope=nn.Parameter(torch.tensor(0.3), requires_grad=False), inplace=True))
    finetune(model, dataloader, finetune_epochs, lr)
    change_module(model, nn.LeakyReLU, _number, nn.LeakyReLU(negative_slope=nn.Parameter(torch.tensor(0.7), requires_grad=False), inplace=True))
    finetune(model, dataloader, finetune_epochs, lr)
    change_module(model, nn.LeakyReLU, _number, nn.LeakyReLU(negative_slope=nn.Parameter(torch.tensor(1), requires_grad=False), inplace=True))
    finetune(model, dataloader, finetune_epochs, lr)

def change_module2(model, old_module, _number, new_module):
    model._modules['non-linearity'] = new_module

def remove_relu2(model, dataloader, old_module, _number, finetune_epochs, lr):
    change_module2(model, old_module, _number, nn.LeakyReLU(negative_slope=nn.Parameter(torch.tensor(0.1), requires_grad=False), inplace=True))
    finetune(model, dataloader, finetune_epochs, lr)
    change_module2(model, nn.LeakyReLU, _number, nn.LeakyReLU(negative_slope=nn.Parameter(torch.tensor(0.3), requires_grad=False), inplace=True))
    finetune(model, dataloader, finetune_epochs, lr)
    change_module2(model, nn.LeakyReLU, _number, nn.LeakyReLU(negative_slope=nn.Parameter(torch.tensor(0.7), requires_grad=False), inplace=True))
    finetune(model, dataloader, finetune_epochs, lr)
    change_module2(model, nn.LeakyReLU, _number, nn.LeakyReLU(negative_slope=nn.Parameter(torch.tensor(1), requires_grad=False), inplace=True))
    finetune(model, dataloader, finetune_epochs, lr)

def remove_relu_imagenet_vgg16(model, relu_id, args):
    num_hinge = args.num_hinging_steps
    #set data loader to device
    dataloader = get_dataloader('imagenet')
    initial_acc = evaluate(model, dataloader['val'], device='cpu')
    # model.cpu()
    del dataloader
    print(f"initial accuracy is {initial_acc:.2f}%")
    for i in range(1, num_hinge + 1):
        print(f"removing relu with negative slope {i * 1/num_hinge}")
        model.classifier._modules[str(relu_id)] = nn.LeakyReLU(negative_slope=nn.Parameter(torch.tensor(i * 1/num_hinge), requires_grad=False), inplace=True)
        run_finetune(args.nprocs, model, args.lr, args.max_finetune_epochs, initial_acc, args)
        model.load_state_dict(torch.load('temp_best_model.pth'))
        # save model
        torch.save(model.state_dict(), args.log_dir + f"/imagenet_vgg16_{relu_id}_{i * 1/num_hinge}.pth")

def collapse_layers_imagenet_vgg16(model, relu_id):
    w0 = model.classifier._modules[str(relu_id - 1)].weight.data
    w3 = model.classifier._modules[str(relu_id + 2)].weight.data
    b0 = model.classifier._modules[str(relu_id - 1)].bias.data
    b3 = model.classifier._modules[str(relu_id + 2)].bias.data
    model.classifier._modules[str(relu_id + 2)].weight.data = torch.matmul(w3, w0)
    model.classifier._modules[str(relu_id + 2)].bias.data = torch.matmul(w3, b0) + b3

    model.classifier._modules[str(relu_id - 1)] = nn.Identity()
    model.classifier._modules[str(relu_id)] = nn.Identity()
    model.classifier._modules[str(relu_id + 1)] = nn.Identity()
    
