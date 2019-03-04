import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import LeNet5

import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
epoches = 10
learning_rate = 0.01
n_classes = 10

data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = LeNet5.create_model()
model = model.to(device)
# choose SGD as optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.5)
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
loss_func = nn.CrossEntropyLoss(size_average = False)

def train():
    model.train() # switch to train mode
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data) # put data into LeNet5, calculate output
        # loss = loss_func(output, label) # loss
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        # put log
        if batch_idx % 50 == 0:
            print("batch_idx = %d, train_loss = %f"%(batch_idx, loss.item()))

def test():
    model.eval() # switch to test mode
    total_loss = 0
    corr = 0
    for (data, label) in test_loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        # total_loss += loss_func(output, label).item() # loss
        total_loss += F.nll_loss(output, label, reduction = 'sum').item()
        _, res = output.max(1)
        corr += res.eq(label.view_as(res)).sum().item()
    total_loss /= len(test_loader.dataset)
    print("%d/%d, acc = %f, test_loss = %f"%(corr, len(test_loader.dataset), corr / len(test_loader.dataset), total_loss))


for n_epoch in range(epoches):
    print('epoch %d:'%(n_epoch + 1))
    train()
    test()

torch.save(model, "./model/model.pth")
torch.save(model.state_dict(), "./model/model_param.pth")