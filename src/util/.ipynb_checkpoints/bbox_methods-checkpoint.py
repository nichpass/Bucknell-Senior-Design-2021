import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import matplotlib as mpl
warnings.filterwarnings("ignore",category=mpl.cbook.mplDeprecation)

import os
import torch
import torchvision

import statistics
import h5py
import copy
import cv2
import deepdish as dd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from sklearn.metrics import roc_auc_score
from skimage import measure

import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import roc_curve, auc, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

import tempfile
import time
import pickle as pkl;

import util.constants as constants
from util.train_methods import gen_state_dict


dis_small_map = {'Cardiomegaly': 0, 'Effusion': 1, 'Mass': 2, 'Nodule': 3, 'Atelectasis': 4}


class ResNet(nn.Module):
    def __init__(self, model_file):
        super(ResNet, self).__init__()
        
        self.model_file = model_file
        
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        self.model.fc = nn.Sequential(*[
            nn.Linear(in_features=512, out_features=6, bias=True),
            nn.Sigmoid()
        ])
        
        self.features_conv = nn.Sequential(*list(self.model.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier =  nn.Sequential(*list(self.model.children())[-2:][1:])
        
        self.gradients = None
        
    def load_weights(self):
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(gen_state_dict(self.model_file))
    
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)

        h = x.register_hook(self.activations_hook)

        x = self.global_pool(x)
        x = x.view(1, -1)    
        x = self.classifier(x)
        
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)
    
    
def check_contains(b1, b2):
    lx = b1['low_x'] <= b2['low_x']
    ly = b1['low_y'] <= b2['low_y']
    hx = b1['high_x'] >= b2['high_x']
    hy = b1['high_y'] >= b2['high_y']
    
    return lx and ly and hx and hy


def check_too_big(b):
    #if box is larger than ~1/3 of the image then get rid of it
    w = b['high_x'] - b['low_x']
    h = b['high_y'] - b['low_y']
    return w * h > 350000


def compute_dist(b, avg):
    # difference in size between average box for disease class and current box
    x = b['high_x'] - b['low_x']
    y = b['high_y'] - b['low_y']
    return abs(x * y - avg)  


def compute_weight_score(b, heatmap, disease_avg):
    # score based on average pixel in heatmap value and distance from average box size
    lx = b['low_x']
    ly = b['low_y']
    hx = b['high_x']
    hy = b['high_y']
    
    area = (hx - lx) * (hy - ly)
    distance = abs(area - disease_avg)
    
    avg_pixel_val = np.sum(heatmap[ly:hy, lx:hx]) / (area)
    score = avg_pixel_val - distance / 1000
    
    return score


def get_bboxes(label_map, heatmap, disease):
    # a variety of heuristics are applied to the label map that contains independent shapes
    uniques = np.unique(label_map)
    boxes = {} 
    
    # Step 1: go through each unique 'blob' and find the edge x and y values to generate bounding boxes
    for u in uniques:
        low_x = 9999
        low_y = 9999
        high_x = -1
        high_y = -1
        
        for r in range(len(label_map)):
            for c in range(len(label_map[0])):
                
                cur = label_map[r][c]
                
                if cur == u:
                    low_x = min(low_x, c)
                    low_y = min(low_y, r)
                    high_x = max(high_x, c)
                    high_y = max(high_y, r)
                    
                    #explore_map = 1
                    
        boxes[u] = {'low_x' : low_x, 'low_y' : low_y,
                        'high_x' : high_x, 'high_y' : high_y}
     
    
    # Step 2: remove the boxes that are too big (larger than 1/3 of the image) 
    boxes_temp = copy.deepcopy(boxes)
    for k, b in boxes.items():
        if check_too_big(b):
            boxes_temp.pop(k, None)
            
    boxes = boxes_temp
    
    # Step 3: select the 10 boxes with the sizes closest to that of the average box size for this disease (based on training data)
    avg_size = constants.BBOX_AVG_SIZE[disease]
    
    box_dist_from_avg = {}
    best_boxes = {}
    for k, b in boxes.items():
        box_dist_from_avg[k] = compute_dist(b, avg_size)
        
    box_dist_from_avg = {k : v for k, v in sorted(box_dist_from_avg.items(), key=lambda x: x[1])}
    
    for i, k in enumerate(box_dist_from_avg.keys(), 1):
        best_boxes[k] = boxes[k]
        
        if i == 10: # take only the best 10
            break
                
    boxes = best_boxes
    
    # Step 4: Select best box using scoring function that combines average pixel value with distance from average box size
    box_weight_score = {}
    best_boxes = {}

    for k, b in boxes.items():
        box_weight_score[k] = compute_weight_score(b, heatmap, avg_size)
                
    box_weight_score = {k : v for k, v in sorted(box_weight_score.items(), key=lambda x: -x[1])}

    for i, k in enumerate(box_weight_score.keys(), 1):
        best_boxes[k] = boxes[k]
        
        if i == 1: # take only the best 1
            break
            
    boxes = best_boxes
    
    # Step 5: reset the keys on the boxes, for ex:  [5, 7, 11] --> [0, 1, 2]
    boxes_temp = {}
    idx = 0
    for k in boxes.keys():
        boxes_temp[idx] = boxes[k]
        idx += 1
        
    boxes = boxes_temp
    return boxes


def convert_to_rgb(np_arr):
    img = np.uint8(np.interp(np_arr, (np_arr.min() * 1/5, np_arr.max()), (0, 255)))
    img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
    b, g, r = cv2.split(img)
    img = cv2.merge([r,g,b])
    
    return img
    
    
def get_grad_map(filename, model, disease):
    img_path = filename # './data/sorted_images/' + data['disease'] + '/' + data['view'] + '/' + data['img_name']
    transforms = [
        torchvision.transforms.ToTensor()]
    img = Image.open(img_path).convert('RGB')
    
    for t in transforms:
        img = t(img)
    
    img = img.unsqueeze(dim=0)
    img = img.cuda();
    
    model.eval()

    pred = model(img)
    pred[0][dis_small_map[disease]].backward()
    
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) 
    activations = model.get_activations(img).detach() 

    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = heatmap.cpu()
    heatmap /= torch.max(heatmap)
    
    grad_activations = heatmap
    
    img = cv2.imread(img_path)
    
    heatmap = cv2.resize(np.float32(heatmap), (img.shape[1], img.shape[0]))
    heatmap = (heatmap - np.min(heatmap))/np.ptp(heatmap) # convert from [-1, 1] to [0, 1]
    heatmap = np.uint8(255 * heatmap)
        
    colormap = cv2.resize(np.float32(grad_activations), (1024, 1024), interpolation=cv2.INTER_CUBIC)
    colormap = convert_to_rgb(colormap)
    
    superimposed_img = colormap * 0.5 + img * 0.5
    
    blobs = heatmap > 140
    all_labels = measure.label(blobs)
        
    return all_labels, heatmap, grad_activations, img


def calc_iobb(filename, model, disease):
    label_map, heatmap, _, _ = get_grad_map(filename, model, disease)
    boxes = get_bboxes(label_map, heatmap, disease)
    
    single_level = np.array(label_map >= 1, dtype='int')
    real_box = np.zeros((1024, 1024), dtype='int')
    
    x = int(data['x'])
    y = int(data['y'])
    width = int(data['width'])
    height = int(data['height'])
    
    for xi in range(x, x + width):
        for yi in range(y, y + height):
            real_box[yi][xi] = 1

    pred_boxes = np.zeros((1024, 1024), dtype='int')
    
    for _, b in boxes.items():
        for xi in range(b['low_x'], b['high_x']+1):
            for yi in range(b['low_y'], b['high_y']+1):
                pred_boxes[yi][xi] = 1
                
    return jaccard_score(real_box.flatten(), pred_boxes.flatten())
    
    
def bbox_main(filename, model, disease):
        
    all_labels, heatmap, grad_activations, img = get_grad_map(filename, model, disease)
    
    grad_img = convert_to_rgb(grad_activations)

    heattest = cv2.resize(np.float32(grad_activations), (1024, 1024), interpolation=cv2.INTER_CUBIC)#, interpolation=cv2.INTER_LANCZOS4)
    heattest = convert_to_rgb(heattest)
    
    boxdict = get_bboxes(all_labels, heatmap, disease)
    
    #just xray and bounding boxes
    fig, ax = plt.subplots()
    ax.imshow(img)
    for i in range(len(boxdict)):
        x = boxdict[i]['low_x']
        y = boxdict[i]['low_y']
        width = boxdict[i]['high_x'] - x
        height = boxdict[i]['high_y'] - y
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='red', facecolor='none' )
        ax.add_patch(rect)
        
    img_name = filename.split('/')[-1]
    extension = img_name.split('.')[-1]        
    
    filepath = 'bbox_' + disease + '_' + img_name
    plt.savefig(filepath)

    print(disease, 'file saved to:', filepath)
    
