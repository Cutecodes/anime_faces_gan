import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn.functional as F

from ANIME import ANIME
from DCGAN import *

# Training settings
batch_size = 200
lr = 0.0002


# MNIST Dataset
# 自行重载数据集
train_dataset = ANIME(root='./anime/',transform=transforms.ToTensor())


# Data Loader

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)


#model 

D = Discriminator()
Doptimizer = optim.Adam(D.parameters(),lr=lr,betas=(0.5,0.999))
Dlossfunc = nn.BCEWithLogitsLoss()

G = Generator()
Goptimizer = optim.Adam(G.parameters(),lr=lr,betas=(0.5,0.999))
Glossfunc = nn.BCEWithLogitsLoss()
def main():
	# 如果模型文件存在则尝试加载模型参数
    if os.path.exists('./Dmodel.pth'):
        try:
            D.load_state_dict(torch.load('./Dmodel.pth'))
            #G.load_state_dict(torch.load('./Gmodel.pth'))
        except Exception as e:
            print(e)
            print("Parameters Error")
    if os.path.exists('./Gmodel.pth'):
        try:
            #D.load_state_dict(torch.load('./Dmodel.pth'))
            G.load_state_dict(torch.load('./Gmodel.pth'))
        except Exception as e:
            print(e)
            print("Parameters Error")
    for epoch in range(1, 200):
        train(epoch,D,Doptimizer,G,Goptimizer,Dlossfunc,Glossfunc,train_loader)
        torch.save(D.state_dict(),'./Dmodel.pth')
        torch.save(G.state_dict(),'./Gmodel.pth')
       
if __name__ == '__main__':
	main()
