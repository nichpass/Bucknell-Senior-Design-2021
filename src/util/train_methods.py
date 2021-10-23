import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import matplotlib as mpl
warnings.filterwarnings("ignore",category=mpl.cbook.mplDeprecation)

import os
import time
import torch

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


from PIL import Image
from sklearn.metrics import roc_auc_score, auc, roc_curve


disease_map = {"Atelectasis" : 0, "Consolidation" : 1, "Infiltration" : 2, "Pneumothorax": 3, "Edema": 4,
               "Emphysema": 5, "Fibrosis": 6, "Effusion" : 7, "Pneumonia" : 8, "Pleural_Thickening" : 9,
               "Cardiomegaly" : 10, "Nodule" : 11, "Mass" : 12, "Hernia" : 13, "No Finding" : 14 }

dis_small_map = {'Cardiomegaly': 0, 'Effusion': 1, 'Mass': 2, 'Nodule': 3, 'Atelectasis': 4, 'No Finding': 5}

dis_bb_map = {'Cardiomegaly': 0, 'Effusion': 1, 'Mass': 2, 'Nodule': 3,
                 'Atelectasis': 4, 'Infiltration': 5, 'Pneumonia' : 6, 'Pneumothorax' : 7, 'No Finding': 8}

class GetLoader(torch.utils.data.Dataset):
    '''
        NOTE: I hardcoded this one a bit, basically splits validation set in half and gives it to test set
    
        Params: data - the data dictionary
                view - the orientation you want to look at
                diseases - the diseases you would like to look at
                num_imgs - the number of images of each disease you would like
                factor - the ratio of training and testing data
                typ - 0 for training, 1 for testing
    '''
    def __init__(self, data, view, diseases, num_imgs, factor, typ, transforms=None):
        
        #private data
        self.root = os.path.join('/bn-hpc/data/DeepMedIA/data/sorted_images')
        self.data = data # dict object
        self.transforms = transforms
        self.len_data = 0
        datalist = []
                
        #Creating the datalist
        for i in range(len(diseases)):                
            
            if len(data[view][diseases[i]]) <= num_imgs: #if the folder has less images than the desired number of images
                if typ == 'train':
                    start = 0
                    end = int(len(data[view][diseases[i]])*factor)
                elif typ == 'test':
                    start = int(len(data[view][diseases[i]])*factor)
                    end = int(len(data[view][diseases[i]])*factor + len(data[view][diseases[i]])*(1-factor) / 2)
                else:
                    start = int(len(data[view][diseases[i]])*factor + len(data[view][diseases[i]])*(1-factor) / 2)
                    end = -1
            else:
                if typ == 'train':
                    start = 0
                    end = int(num_imgs*factor)
                elif typ == 'test':
                    start = int(num_imgs*factor)
                    end = int(num_imgs*factor + num_imgs*(1-factor) / 2)
                else:
                    start = int(num_imgs*factor + num_imgs*(1-factor) / 2)
                    end = num_imgs
                    
            #print('disease: ', diseases[i], 'num images used: ', min(len(data[view][diseases[i]]), num_imgs))
                        
            datalist.append(self.data[view][diseases[i]][start:end])
        
        for item in datalist:
            self.len_data += len(item)
        
        
        self.img_paths = []
        self.img_labels = []
        
        for dis in datalist:
            for data in dis:
                #creating the image path
                data['img_path'] = os.path.join(self.root, data['classes'][0], view, data['img_name'])            
                diseases_item = data['classes']

                one_hot = np.zeros(9)
                for d in diseases_item:
                    if d in diseases:
                        hot_index = dis_bb_map[d]
                        one_hot[hot_index] = 1

                self.img_paths.append(data['img_path'])
                self.img_labels.append(torch.Tensor(one_hot))


    def __getitem__(self, item):
        
        img_path, img_label = self.img_paths[item], self.img_labels[item]
  
        # TODO: fix this hot fix -> recreate data object with underscore in name
        img_path = img_path.replace('No Finding', 'No_Finding')
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img_path = img_path.replace('/AP/', '/PA/')
            img = Image.open(img_path).convert('RGB')
            
        self.cur_img_path = img_path

        if self.transforms is not None:
            for t in self.transforms:
                img = t(img)

        return img, img_label
    
    def get_img_path(self):
        return self.cur_img_path

    def __len__(self):
        return self.len_data
    
    
def load_old_weights(net, path):
    try:
        net.load_state_dict(torch.load(path, verbose=False))
        print('successfully loaded')
    except:
        print('issue with weight names, needs manual fix')
        

def gen_state_dict(model_path):
    state_dict = torch.load(model_path)
    
    n_prefix = 0
    first_key = list(state_dict.keys())[0]
    words = first_key.split('.')
    for w in words:
        if w == 'module':
            n_prefix += 1
            
    if n_prefix == 0:
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            new_state_dict['module.' + k] = state_dict[k]
        return new_state_dict        

    elif n_prefix > 1:
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            new_k = k[7 * n_prefix:]
            new_state_dict[new_k] = state_dict[k]
        return new_state_dict 
    
    else:
        return state_dict
        
## source: https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
def hard_binary_accuracy(batch, labels):
    batch = torch.round(batch)
    confusion_matrix = batch / labels
    
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_matrix = batch / labels
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_matrix == 1).item()
    false_positives = torch.sum(confusion_matrix == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_matrix)).item()
    false_negatives = torch.sum(confusion_matrix == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def train(epoch, network, train_loader, optimizer, batch_size):

    network.train()
    train_losses = []
    num_tested = 0
    net_loss = 0
    
    total_true_pos, total_false_pos, total_true_neg, total_false_neg = 0, 0, 0, 0
    
    for batch_idx, (data, target) in enumerate(train_loader):  # (output - actual )
        
        if torch.cuda.is_available:
            data, target = data.cuda(), target.cuda()

        output = network(data)
        
        #criterion = torch.nn.BCELoss(weight=torch.Tensor(class_weights).cuda())  # 1 0 1 0 0 0 
        criterion = torch.nn.BCELoss()
        loss = criterion(output, target)
        loss.backward()
        
        net_loss += loss.clone().detach().cpu().item()
        train_losses.append(loss.item())

        num_tested += len(data)
        
        #if batch_idx % 1 == 0 or batch_idx == len(train_loader) - 1:
        optimizer.step()
        optimizer.zero_grad()

        pred = output.clone().detach().clone().cpu()
        target = target.cpu()

        true_pos, false_pos, true_neg, false_neg = hard_binary_accuracy(pred, target)
        total_true_pos += true_pos
        total_false_pos += false_pos
        total_true_neg += true_neg
        total_false_neg += false_neg
            
    net_loss /= (batch_size * len(train_loader))
    return net_loss, total_true_pos, total_false_pos, total_true_neg, total_false_neg

def test(network, test_loader, batch_size):
    
    network.eval()
    test_losses = []
    net_loss = 0    

    total_true_pos, total_false_pos, total_true_neg, total_false_neg = 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available:
                data, target = data.cuda(), target.cuda()
            
            output = network(data)
            
            #criterion = torch.nn.BCELoss(weight=torch.Tensor(class_weights).cuda())
            criterion = torch.nn.BCELoss()

            net_loss += criterion(output, target).item()

            pred = output.detach().clone().cpu()
            target = target.cpu()
            
            true_pos, false_pos, true_neg, false_neg = hard_binary_accuracy(pred, target)
            total_true_pos += true_pos
            total_false_pos += false_pos
            total_true_neg += true_neg
            total_false_neg += false_neg
            
        
    net_loss /= (batch_size * len(test_loader))
    return net_loss, total_true_pos, total_false_pos, total_true_neg, total_false_neg


def display_roc_curve(loader, model, dis_map, batch_size):
    preds = {}
    labels = {}

    for imgs, labs in loader:
        p = model(imgs)
        
        p = p.detach().cpu().numpy()
        labs = labs.detach().cpu().numpy()
        
        for pi, li in zip(p, labs):
            for name, idx in dis_map.items():
                preds[name] = np.append(preds.get(name, np.array([])), pi[idx])
                labels[name] = np.append(labels.get(name, np.array([])), li[idx])
    
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    
    best_thresholds = {}
    
    for name in dis_map: #preds.keys():
        y_test = labels[name]
        y_score = preds[name]

        fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        best_id = np.argmax(tpr - fpr)
        #print('best vals:', np.max(tpr - fpr))
        best_thresholds[name] = thresholds[best_id]

        plt.plot(fpr, tpr, lw=1, label= name + ' ROC curve (area = %0.2f)' % roc_auc)
        

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc="lower right")
    plt.show()
    
    del preds
    del labels
    
    #print('best thresholds:')
    #print(best_thresholds)
    

