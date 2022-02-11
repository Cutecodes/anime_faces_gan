import matplotlib.pyplot as plt
import torch
import os
from DCGAN import *
import pickle
import numpy

if os.path.exists("images"):
        with open("images","rb+") as f:
            listp = pickle.load(f)
            
            pickle.dump(listp,f)

for (cnt,i) in enumerate(listp):

    ax = plt.subplot(4, 4, cnt+1)
    # ax.axis('off')
    i = i.detach().numpy()
    i = i/2 +0.5
    i = np.transpose(i,(1,2,0))
    ax.imshow(i)
    if cnt==16:
        break
    plt.pause(1)
