#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import matplotlib as mpl
warnings.filterwarnings("ignore",category=mpl.cbook.mplDeprecation)

import os
import torch
import torchvision
import statistics
import copy
import cv2

from PIL import Image
from skimage import measure

import sklearn.metrics as metrics
import numpy as np

import pickle as pkl;

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import util.constants as constants
from util.bbox_methods import (ResNet,
                               bbox_main,
                               gen_state_dict
                              )


# In[2]:


model_path = './saved_models/6_disease_multi_label.pth'
#img_path = './beta_test_imgs/00000013_030.png'
transform = torchvision.transforms.ToTensor()

dis_pred_map = {'Cardiomegaly': 0, 'Effusion': 1, 'Mass': 2, 'Nodule': 3, 'Atelectasis': 4}
has_disease_dict = {'Cardiomegaly': False, 'Effusion': False, 'Mass': False, 'Nodule': False, 'Atelectasis': False}


# In[3]:


class_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)

class_model.fc = nn.Sequential(*[
    nn.Linear(in_features=512, out_features=6, bias=True),
    nn.Sigmoid(),
])

class_model = torch.nn.DataParallel(class_model)
class_model.cuda();

class_model.load_state_dict(gen_state_dict(model_path))


# In[4]:
img_path = input('Enter the relative path to your chest x-ray image: ')

img = Image.open(img_path).convert('RGB')
img = transform(img).unsqueeze(0)


# In[5]:


class_model.eval()

with torch.no_grad():
    pred = class_model(img).cpu().numpy()[0]


# In[6]:


for d in has_disease_dict.keys():
    pred_index = dis_pred_map[d]
    has_disease_dict[d] = pred[pred_index] > constants.BEST_THRESHOLD[d]


# In[7]:


print('\n---Results---')

for disease, status in has_disease_dict.items():
    print(('{0:15} :').format(disease), status)

print('\nGenerating bounding boxe images...')
# In[8]:


gmodel = ResNet(model_path)
gmodel.load_weights()


# In[9]:


for disease, disease_present in has_disease_dict.items():
    if disease_present:
        bbox_main(img_path, gmodel, disease)


print('\nImage creation complete.')
# In[ ]:




