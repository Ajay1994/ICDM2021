from __future__ import print_function, division
from tqdm import tqdm
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from augmentation import ColourDistortion, BlurOrSharpen

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
from resnet import *


EVAL_IMAGE_LIST = "test.txt"
DATA_DIR = "/data/NIH_Xray/images/"
torch.cuda.init()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'evaluate': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
train_sampler = None
batch_size = 64
workers = 4
N_CLASSES = 8
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax']

# N_CLASSES = 14
# CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
#                 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

eval_dataset = ChestXrayDataSet(data_dir = DATA_DIR, image_list_file = EVAL_IMAGE_LIST, transform = data_transforms["evaluate"])

eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)
dataloaders = {"evaluate": eval_loader}


class Counter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_AUCs(gt, pred):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def compute_Accuracy(image_name, gt, pred):
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    acc_count = 0
    for i in range(0, len(gt_np)):
        if gt_np[i] == pred_np[i]:
            acc_count += 0
    return acc_count/len(gt_np)
    
def eval_model(model):
    since = time.time()
    losses = Counter()
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    losses.reset()

    model.eval()   # Set model to evaluate mode
    with torch.no_grad():
        t = tqdm(enumerate(dataloaders["evaluate"]),  desc='Loss: **** ', total=len(dataloaders["evaluate"]), bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (image_name, inputs, labels) in t:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            gt = torch.cat((gt, labels), 0)
            pred = torch.cat((pred, outputs.data), 0)
        AUCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUCs).mean()
    print("Average AUC : {}".format(AUROC_avg))
    for i in range(N_CLASSES):
        print('{} \t {}'.format(CLASS_NAMES[i], AUCs[i]))


def initialize_model(model_ft):
    checkpoint = torch.load("RESNET50_PCAM_IMAGENET_model_best.pth.tar")
    pretrained_dict = checkpoint['state_dict']
    model_dict = model_ft.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model_ft.load_state_dict(model_dict)
    return model_ft

    
model_ft =  ResidualNet('ImageNet', 50, 8, "TripletAttention")
model_ft.load_state_dict(torch.load("./checkpoint/model_BCE_837.pth"))
print(model_ft)
print("Model Weight Loaded")
model_ft = model_ft.to(device)
print("Model transferred to device")
criterion = nn.BCEWithLogitsLoss()

eval_model(model_ft)