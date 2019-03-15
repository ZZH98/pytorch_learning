import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import sys
import os

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
epoches = 30
learning_rate = 0.001
n_classes = 10

save_path = "./model/model.pth"

os.environ['CUDA_VISIBLE_DEVICES'] = '12,13,14,15'

data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

# cifar dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=data_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=data_transform, download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = models.vgg16(pretrained = False, num_classes = n_classes)
model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.5)
loss_func = nn.CrossEntropyLoss(size_average = False)

def train():
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        output = model(data) # put data into net
        # loss = loss_func(output, label) # loss
        output = F.log_softmax(output, dim = 1)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        # put log
        if batch_idx % 50 == 0:
            print("batch_idx = %d, train_loss = %f"%(batch_idx, loss.item()))

def test():
    model.eval()
    total_loss = 0
    corr = 0
    for (data, label) in test_loader:
        torch.cuda.empty_cache()
        data = data.cuda()
        label = label.cuda()
        output = model(data)
        # total_loss += loss_func(output, label).item() # loss
        output = F.log_softmax(output, dim = 1)
        total_loss += F.nll_loss(output, label).item()
        _, res = output.max(1)
        corr += res.eq(label.view_as(res)).sum().item()
    total_loss /= len(test_loader.dataset)
    print("%d/%d, acc = %f, test_loss = %f"%(corr, len(test_loader.dataset), corr / len(test_loader.dataset), total_loss))

for n_epoch in range(epoches):
    print('epoch %d:'%(n_epoch + 1))
    train()
    test()

torch.save(model.state_dict(), save_path)
