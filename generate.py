import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from DCGAN import *
import pickle



def generate():
    G = Generator()
    G.load_state_dict(torch.load('./Gmodel.pth'))
    ginputdata = Variable(torch.randn((50,1,100)))
    goutput = G(ginputdata)
    if os.path.exists("images"):
        with open("images","rb+") as f:
            list = pickle.load(f)
            list += goutput
            pickle.dump(list,f)
    else:
        list = []
        list +=goutput
        with open("images","wb+") as f:
            pickle.dump(list,f)

        

    for (cnt,i) in enumerate(goutput):

        ax = plt.subplot(4, 4, cnt+1)
        # ax.axis('off')

        i = i/2+0.5
        i = i.detach().numpy()
        i = np.transpose(i,(1,2,0))
        ax.imshow(i)
        if cnt==15:
            break
        plt.pause(1)

def main():
    generate()

if __name__ == '__main__':
    main()
