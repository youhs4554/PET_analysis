#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import glob
import tqdm
from natsort import natsorted
import os, sys, random
from imblearn.over_sampling import SMOTE
import skimage
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from IPython.core.debugger import set_trace


# In[ ]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="5"


# In[ ]:


cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')


# In[ ]:


image_prepro = transforms.Compose([
    transforms.Resize(128),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

def save_patches(frame_root, save_root, patch_size, global_ix,
                 seed, n_sample=100):
    if not os.path.exists(save_root):
        os.system(f'mkdir -p {save_root}')
    
    filenames = natsorted(os.listdir(frame_root))

    volume = []
    for fn in filenames:
        fn = os.path.join(frame_root, fn)
        #img = skimage.io.imread(fn, plugin='tifffile')
        img = image_prepro(Image.open(fn)).numpy()[0]
        volume.append(img)

    volume = np.array(volume)
    
    # random volume cropping for 100 times
    random.seed(seed)
    c = 0
    while c < n_sample:
        rand_z, rand_y, rand_x = [ np.random.randint(0, volume.shape[i]-patch_size) for i in range(3) ]
        crop = volume[rand_z:rand_z+patch_size, rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
        if all([ len(np.unique(s)) > 0.2*(patch_size**2) for s in crop]):
            np.save(os.path.join(save_root, f'patch_{global_ix}.npy'), crop)
            global_ix += 1
            c += 1
    
    return global_ix
    
def generate_patches(root, save_root,
                    patch_size=7, train=True):
    dirpath = np.array(natsorted(glob.glob(root+'/*/*')))

    train_size = int(len(dirpath)*0.8)
    test_size = len(dirpath)-train_size

    np.random.seed(999)
    train_ixs = np.random.choice(range(len(dirpath)), size=train_size, replace=False)
    test_ixs = np.array(
        list(set(range(len(dirpath))) - set(train_ixs))
    )

    if train:
        dirpath = dirpath[train_ixs]
    else:
        dirpath = dirpath[test_ixs]
                
    suffix = 'train' if train else 'test'
                
    save_root = os.path.join(save_root, suffix)
    
    global_ix = 0
    for frame_root in tqdm.tqdm(dirpath):
        seed = random.randint(-sys.maxsize, sys.maxsize)
        global_ix = save_patches(frame_root, save_root, patch_size, global_ix=global_ix, seed=seed, n_sample=100)


# # Generate Patch Data

# In[ ]:


root = '/data/DSMC_breast468/full'


# In[ ]:


if not os.path.exists('/data/DSMC_breast468/patches'):
    generate_patches(root, save_root='/data/DSMC_breast468/patches', train=True)  # trainset
    generate_patches(root, save_root='/data/DSMC_breast468/patches', train=False) # testset


# # Patch Datasets

# In[ ]:


class PETImagePatchDataset(Dataset):
    def __init__(self, root, patch_size=7, train=True):
        suffix = 'train' if train else 'test'
        root = os.path.join(root, suffix)
        
        self.dirpath = np.array(natsorted(glob.glob(root+'/*')))
                
    def __len__(self):
        return len(self.dirpath)
    
    def __getitem__(self, ix):
        volume = np.load(self.dirpath[ix])
        return torch.FloatTensor(volume).flatten()


# In[ ]:


root = '/data/DSMC_breast468/patches'
datasets = {
    split : PETImagePatchDataset(root, train=(split=='train'))
    for split in ['train', 'test']
}
dataloaders = {
    split: DataLoader(datasets[split], batch_size=128, shuffle=(split=='train'))
    for split in ['train', 'test']
}


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import os, datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from torch import nn
from torch.autograd import Variable
from networks import SparseAutoencoderKL

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

def kl_divergence(p, p_hat):
    funcs = nn.Sigmoid()
    p_hat = torch.mean(funcs(p_hat), 1)
    p_tensor = torch.Tensor([p] * len(p_hat)).to(device)
    return torch.sum(p_tensor * torch.log(p_tensor) - p_tensor * torch.log(p_hat) + (1 - p_tensor) * torch.log(1 - p_tensor) - (1 - p_tensor) * torch.log(1 - p_hat))

def sparse_loss(autoencoder, images):
    loss = 0
    values = images
    for i in range(3):
        fc_layer = list(autoencoder.encoder.children())[2 * i]
        relu = list(autoencoder.encoder.children())[2 * i + 1]
        values = fc_layer(values)
        loss += kl_divergence(DISTRIBUTION_VAL, values)
    for i in range(2):
        fc_layer = list(autoencoder.decoder.children())[2 * i]
        relu = list(autoencoder.decoder.children())[2 * i + 1]
        values = fc_layer(values)
        loss += kl_divergence(DISTRIBUTION_VAL, values)
    return loss

def model_training(autoencoder, train_loader, epoch):
    loss_metric = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    autoencoder.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        images = data
        images = Variable(images)
        images = images.view(images.size(0), -1)
        if cuda: images = images.to(device)
        outputs = autoencoder(images)
        mse_loss = loss_metric(outputs, images)
        kl_loss = sparse_loss(autoencoder, images)
        loss = mse_loss + kl_loss * SPARSE_REG
        loss.backward()
        optimizer.step()
        if (i + 1) % LOG_INTERVAL == 0:
            print('Epoch [{}/{}] - Iter[{}/{}], Total loss:{:.4f}, MSE loss:{:.4f}, Sparse loss:{:.4f}'.format(
                epoch + 1, EPOCHS, i + 1, len(train_loader.dataset) // BATCH_SIZE, loss.item(), mse_loss.item(), kl_loss.item()
            ))

def evaluation(autoencoder, test_loader):
    total_loss = 0
    loss_metric = nn.MSELoss()
    autoencoder.eval()
    for i, data in enumerate(test_loader):
        images = data
        images = Variable(images)
        images = images.view(images.size(0), -1)
        if cuda: images = images.to(device)
        outputs = autoencoder(images)
        loss = loss_metric(outputs, images)
        total_loss += loss * len(images)
    avg_loss = total_loss / len(test_loader.dataset)

    print('\nAverage MSE Loss on Test set: {:.4f}'.format(avg_loss))

    global BEST_VAL
    if TRAIN_SCRATCH and avg_loss < BEST_VAL:
        BEST_VAL = avg_loss
        torch.save(autoencoder.state_dict(), './history/sparse_autoencoder_KL.pt')
        print('Save Best Model in HISTORY\n')


# In[ ]:


EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LOG_INTERVAL = 100
DISTRIBUTION_VAL = 0.3
SPARSE_REG = 1e-3
TRAIN_SCRATCH = True        # whether to train a model from scratch
BEST_VAL = float('inf')     # record the best val loss

#train_loader, test_loader = data_utils.load_mnist(BATCH_SIZE)
train_loader, test_loader = dataloaders['train'], dataloaders['test']

autoencoder = SparseAutoencoderKL()
if cuda: autoencoder.to(device)

if TRAIN_SCRATCH:
    # Training autoencoder from scratch
    for epoch in range(EPOCHS):
        starttime = datetime.datetime.now()
        model_training(autoencoder, train_loader, epoch)
        endtime = datetime.datetime.now()
        print(f'Train a epoch in {(endtime - starttime).seconds} seconds')
        # evaluate on test set and save best model
        evaluation(autoencoder, test_loader)
    print('Trainig Complete with best validation loss {:.4f}'.format(BEST_VAL))



