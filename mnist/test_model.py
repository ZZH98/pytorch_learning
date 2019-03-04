import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import LeNet5
from PIL import Image

import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([transforms.Resize(28), 
                                    transforms.Grayscale(num_output_channels = 1), 
                                    transforms.ToTensor(), 
                                    ])

model = LeNet5.create_model()
model.load_state_dict(torch.load('./model/model_param.pth'))
# model = torch.load('./model/model.pth')
model = model.to(device)

test_dir = './test_images/'

imgs = os.listdir(test_dir)
imgnum = len(imgs)
for i in range(imgnum):
    print(imgs[i])
    img = Image.open(test_dir + imgs[i])
    img = data_transform(img)
    img.unsqueeze_(dim = 0)
    img = img.to(device)
    output = model(img)
    _, res = output.max(1)
    print(res[0].item())