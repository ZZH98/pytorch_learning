import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

#inherit from nn.Module
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self ).__init__() 
        self.conv1 = nn.Conv2d(1, 6, 5, padding = 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x): # forward is automatically called when execute y = model(x), x is a batch of data
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #print('size', x.size())   # 256, 16, 5, 5
        x = x.view(x.size(0), -1)
        #print('size', x.size()) # 256, 400
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        # return x

def create_model():
    model = LeNet5()
    return model