#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
import distiller
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time

classes = [i for i in range(1000)]

def countParameters(net):
    params = 0
    for par in net.parameters():
        k = 1
        for x in par.size():
            k *= x
        params += k
    return params

def checkAccuracy(model, device, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    print('Accuracy of the network on the test images: %.2f %%' % (
        accuracy))
    model.train()
    return accuracy


def testModel(net, device, testloader):
    global classes
    correct = 0
    total = 0
    net.eval()
    print('Parameters:', countParameters(net))
    t1 = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    t2 = time.time()
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
        100 * correct / total))
    print('Average Latency for 10000 test images:', (t2-t1)/10000,'seconds')
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(10):
        print('Accuracy of %5s : %2f %%' % (
            classes[i], 100.0 * class_correct[i] / class_total[i]))


def trainModel(net, modelLocation, device, trainloader, testloader, opt, startEpoch, totalEpochs, accuracy = 0):
    
    criterion = nn.CrossEntropyLoss()
    bestAccuracy = accuracy
    bestEpoch = startEpoch
    torch.save(net.state_dict(), modelLocation)
    if opt == optim.SGD:
        scheme = 1
    else:
        scheme = 0
    for epoch in range(startEpoch, totalEpochs):  # loop over the dataset multiple times
        if scheme == 1:
            if epoch < 150:
                optimizer = opt(net.parameters(), lr=0.1, momentum = 0.9, weight_decay=5e-4)
            elif epoch < 250:
                optimizer = opt(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = opt(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = opt(net.parameters(), lr=0.001, weight_decay=5e-4)
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            # print("inputs.shape", inputs.shape)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            
            if i % 50 == 49:    # print every 128 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 128))
                running_loss = 0.0
        
        accuracy = checkAccuracy(net, device, testloader)
        if accuracy >= bestAccuracy:    
            torch.save(net.state_dict(), modelLocation)
            bestAccuracy = accuracy
            bestEpoch = epoch+1
        print('Best Accuracy of', bestAccuracy,'at epoch',bestEpoch)
        
    print('Finished Training Model.')
    try:
        net.load_state_dict(torch.load(modelLocation))
    except:
        pass
    
    testModel(net, device, testloader)


def trainModelKD(model, modelLocation, teacher, device, trainloader, testloader, alpha, T, opt, startEpoch, totalEpochs, accuracy = 0):
    
    criterion = nn.CrossEntropyLoss()
    dlw = distiller.DistillationLossWeights(alpha*T*T, 1-alpha, 0.0)
    kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, T, dlw)
    kd_policy.active = True
    bestAccuracy = accuracy
    bestEpoch = startEpoch
    torch.save(model.state_dict(), modelLocation)
    if opt == optim.SGD:
        scheme = 1
    else:
        scheme = 0
    print(f"model has {get_num_parameters(model, True)/1e6:.2f} M parameters")
    for epoch in range(startEpoch, totalEpochs):  # loop over the dataset multiple times
        start = time.time()
        if scheme == 1:
            if epoch < 150:
                optimizer = opt(model.parameters(), lr=0.1, momentum = 0.9, weight_decay=5e-4)
            elif epoch < 250:
                optimizer = opt(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = opt(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = opt(model.parameters(), lr=0.001, weight_decay=5e-4)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            output = kd_policy.forward(inputs)
            loss = criterion(output, labels)
            loss = kd_policy.before_backward_pass(model, epoch, None, None, loss, None).overall_loss        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
        accuracy = checkAccuracy(model, device, testloader)
        if accuracy > bestAccuracy:
            torch.save(model.state_dict(), modelLocation)
            bestAccuracy = accuracy
            bestEpoch = epoch+1
        print('Best Accuracy of', bestAccuracy,'at epoch',bestEpoch)
        end = time.time()
        print("time taken:", end-start)
    
    print('Finished Training Student.')
    try:
        model.load_state_dict(torch.load(modelLocation))
    except:
        pass
    
    testModel(model, device, testloader)



import sys
import vgg
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from utils import *


def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform
    
    return platforms[sys.platform]

models = ['VGG16']

def createResnet(ResType):
    global resDict    
    f = resDict[ResType]
    return f()

def createVGG(VGGType):
    return vgg.VGG(VGGType)

def create5_CNN():
    return cnn5.CNN_5()

def createNet(modelType):
    global models
    global jointModels
    
    for i in range(len(models)):
        if modelType.lower() == models[i].lower():
            if i <= 4:
                return createResnet(models[i])
            elif i <= len(models)-2:
                return createVGG(models[i])
            else:
                return create5_CNN()
    
    if modelType.lower() == 'avg':
        return avNet.AvNet()
    
    if modelType.lower() == 'jn':
        return netJoin.jointNet()
    
    return None


trainModel = trainModel
trainModelKD = trainModelKD
testModel = testModel

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA Device Detected. GPU will be used.")
else:
    device = torch.device("cpu")
    print("No CUDA supported GPU detected. CPU will be used.")

print("Dataset- ImageNet")

if get_platform() == 'Windows':
    workers = 0
else:
    workers = 2

dataloader = get_dataloader("imagenet", batch_size=256)
trainloader = dataloader['train']
testloader = dataloader['val']
opt = optim.Adam
totalEpochs = 500
startEpoch = 0
batchSize = 256

import torchvision.models as models
base_model = models.vgg16(pretrained=True).cuda()
torch.save(base_model.state_dict(), "base_vgg16.pt")

try:
    alpha = 0.5
    T = 2.0
except:
    print('Invalid Arguments for temperature or alpha. Aborting')
    # return
student_model = createVGG('VGG16')
print(student_model)
if student_model is None:
    print('Invalid Model Type. Aborting.')
    # return
student_model = student_model.to(device)    
try:
    student_model.load_state_dict(torch.load("student_vgg.pt"))
    accuracy = checkAccuracy(student_model, device, testloader)
except Exception as e:
    print("Exception loading student model:", str(e))
    accuracy = 0

teacher = base_model
if teacher is None:
    print('Invalid Model Type for Teacher. Aborting.')
    # return
teacher = teacher.to(device)
try:
    teacher.load_state_dict(torch.load("base_vgg16.pt"))
except Exception as e:
    print("Exception loading teacher model:", str(e))
    print('The Teacher model does not exists. Aborting.')
    # return
for param in teacher.parameters():
    param.requires_grad = False
teacher.eval()
trainModelKD(student_model, "student_vgg.pt", teacher, device, trainloader, testloader, alpha, T, opt, startEpoch, totalEpochs, accuracy)