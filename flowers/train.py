import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

'''
the dataset "17flowers" can be downloaded from http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 16
epoches = 50
learning_rate = 0.001
n_classes = 17

save_path = "./model/model.pth"
# data_transform = transforms.Compose([transforms.Resize((30, 30)), transforms.Grayscale(1), transforms.ToTensor(), ])
data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

train_dataset = torchvision.datasets.ImageFolder('./train', transform = data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = torchvision.datasets.ImageFolder('./test', transform = data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

model = models.vgg16(pretrained = False, num_classes = n_classes)
model = model.to(device)
# choose SGD as optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
loss_func = nn.CrossEntropyLoss(size_average = False)

def train():
    model.train() # switch to train mode
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data) # put data into LeNet5, calculate output
        loss = loss_func(output, label) # loss
        # loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        # put log
        if (batch_idx % 10 == 0): 
            print("batch_idx = %d, train_loss = %f"%(batch_idx, loss.item()))

def test():
    model.eval() # switch to test mode
    total_loss = 0
    corr = 0
    for (data, label) in test_loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        total_loss += loss_func(output, label).item() # loss
        # total_loss += F.nll_loss(output, label)
        _, res = output.max(1)
        corr += res.eq(label.view_as(res)).sum().item()
    total_loss /= len(test_loader.dataset)
    print("%d/%d, acc = %f, test_loss = %f"%(corr, len(test_loader.dataset), corr / len(test_loader.dataset), total_loss))


for n_epoch in range(epoches):
    print('epoch %d:'%(n_epoch + 1))
    train()
    test()

torch.save(model.state_dict(), save_path)
