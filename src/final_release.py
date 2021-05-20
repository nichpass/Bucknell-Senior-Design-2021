#!/usr/bin/env python
# coding: utf-8

# In[4]:

import mrcnn.model as modellib
from mrcnn import visualize

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


import random
import math
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.utils import shuffle





# In[5]:

img_path = input('Welcome: please input the relative or absolute path to your chest X-Ray image: ')

model_path = './saved_models/bbox_disease_classifier.pth'
# img_path = './beta_test_imgs/00004461_016.png'
transform = torchvision.transforms.ToTensor()

dis_pred_map =  {'Cardiomegaly': 0, 'Effusion': 1, 'Mass': 2, 'Nodule': 3,
                 'Atelectasis': 4, 'Infiltration': 5, 'Pneumonia' : 6, 'Pneumothorax' : 7, 'No Finding': 8}

has_disease_dict = {dis: False for dis in dis_pred_map.keys()}

IMG_ID = img_path.split('/')[-1]
# In[6]:


class_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)

class_model.fc = nn.Sequential(*[
    nn.Linear(in_features=512, out_features=9, bias=True),
    nn.Sigmoid(),
])

class_model = torch.nn.DataParallel(class_model)
class_model.cuda();

class_model.load_state_dict(gen_state_dict(model_path))


# In[7]:


img = Image.open(img_path).convert('RGB')
img = transform(img).unsqueeze(0)


# In[8]:


class_model.eval()

with torch.no_grad():
    pred = class_model(img).cpu().numpy()[0]


# In[9]:


for d in has_disease_dict.keys():
    pred_index = dis_pred_map[d]
    has_disease_dict[d] = pred[pred_index] > constants.BEST_THRESHOLD[d]


# In[10]:


# In[11]:

print('1')


print(1.1)



# In[12]:
print('dc')
from util.DetectorConfig import DetectorConfig


# In[13]:


#IMG_ID = '00004461_016.png'


# In[14]:

print('2')

#from mrcnn.config import Config
#from mrcnn import utils

#from mrcnn.model import log


print('3')

# In[15]:


ORIG_SIZE = 1024


# In[16]:



class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 9
    
    MAX_GT_INSTANCES = 2
    
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78
    DETECTION_NMS_THRESHOLD = 0.01

inference_config = InferenceConfig()

model_path = './saved_models/mrcnn_final_model.h5'

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=model_path)

# Load trained weights (fill in path to trained weights here)

print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[17]:


OUTPUT_DIR = './mrcnn_out_img/'


# In[19]:


image = np.array(Image.open(img_path).convert('RGB'))
results = model.detect([image])
r = results[0]

ids = np.full(r['class_ids'].shape, 1)
labels = np.full(9, 'Area of Interest')
scores = np.full(r['class_ids'].shape, '')
ax = plt.subplot(1,1,1)

colors=[]
for class_id in ids:
    colors.append((.941, .204, .204))

visualize.display_instances(image, r['rois'], r['masks'], ids, labels, scores, colors=colors, ax=ax)


outfile_name = OUTPUT_DIR+'/mrcnn_out_'+IMG_ID+'.png'
plt.savefig(outfile_name)



print('\n --- RESULTS ---\n')

for disease, status in has_disease_dict.items():
    print(('{0:15} :').format(disease), status)
    
print('\nBounding box image saved to: ' + outfile_name)

# In[ ]:




