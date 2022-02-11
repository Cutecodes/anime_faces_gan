from torch.utils.data import Dataset
import torch
import os
from torchvision.transforms import transforms
import numpy as np
from PIL import Image


class ANIME(Dataset):#need implement __len__(),__getitem__():
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.images = []        
        i = 0
        for filename in os.listdir(self.root):
            im = Image.open(self.root+filename)
            self.images.append(im)
            i += 1
            if i ==2000:
                break
            

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img = self.images[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        img = (img-0.5)*2
        #[-1,1]
        return img

def main():
    import matplotlib.pyplot as plt
    train = ANIME(root='./anime/',transform=transforms.ToTensor())
    print(len(train))
    for (cnt,i) in enumerate(train):
        print(i)
        ax = plt.subplot(4, 4, cnt+1)
        # ax.axis('off')
        i = i.numpy()
        
        i = np.transpose(i,(1,2,0))
        ax.imshow(i)
        plt.pause(0.001)
        if cnt ==15:
            break
    
if __name__ == '__main__':
	main()
