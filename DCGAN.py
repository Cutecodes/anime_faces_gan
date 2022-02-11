import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self):
        super (Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=2,padding=1), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.25),
            #nn.BatchNorm2d(64)# out_channels
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(128)
        )
        self.fc = nn.Linear(128*16,1)


    def forward(self,x):
        in_size = x.size(0)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.conv4(x)
        #print(x.shape)

        x = x.view(in_size,-1)
        #print(x.size)

        x = self.fc(x)

        return x

class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self):
        super (Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100,128*16*16)
        )

        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128,3,stride = 1,padding=1),
            nn.BatchNorm2d(128,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,3,3,stride=1,padding=1),
            nn.Tanh()
        )

    
        

    def forward(self,x):
        
        x = self.fc(x)
        in_size = x.size(0)
        #print(x.shape)
        x = x.view(in_size,128,16,16)
        x = self.conv(x)
        return x

def train(epoch,Dmodel,Doptimizer,Gmodel,Goptimizer,Dlossfunc,Glossfunc,data_loader):
    Dloss_meter = []
    Gloss_meter = []
    f = open("loss",'a')
    for batch_idx,data in enumerate(data_loader):

        # discriminator
        batch_size = len(data)
        noise = Variable(torch.randn((batch_size,1,100)))
        fake_imgs = Gmodel(noise).detach()
        real_imgs = Variable(data)

        real_label = Variable(torch.zeros((batch_size,1)))
        fake_label = Variable(torch.ones((batch_size,1)))
        

        fake_loss = Dlossfunc(Dmodel(fake_imgs),fake_label)
        real_loss = Dlossfunc(Dmodel(real_imgs),real_label)

        Dloss = (fake_loss + real_loss )/2
        Doptimizer.zero_grad() #梯度清零
        Dloss.backward()       #反向传播
        Doptimizer.step()      #使用optimizer进行梯度下降
        Dloss_meter.append(Dloss)

        
        # generator
        noise = Variable(torch.randn((batch_size,1,100)))
        fake_imgs = Gmodel(noise)

        GLoss = Glossfunc(Dmodel(fake_imgs),real_label)
        Goptimizer.zero_grad()
        Doptimizer.zero_grad()
        GLoss.backward()
        Goptimizer.step()

        Dloss_meter.append(Dloss)
        Gloss_meter.append(GLoss)


        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.3f}%)]\tDLoss: {:.6f}\tGLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx * batch_size / len(data_loader.dataset), Dloss.data,GLoss.data))
            f.write("%s\t%s\n"%(str(sum(Dloss_meter)/len(Dloss_meter)),str(sum(Gloss_meter)/len(Gloss_meter))))
        
    f.close()



