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

TRAIN_IMAGE_LIST = "train.txt"
VAL_IMAGE_LIST = "test.txt"
DATA_DIR = "/data/NIH_Xray/images/"

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ColourDistortion(s=0.5),
        BlurOrSharpen(radius=1.),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
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

train_dataset = ChestXrayDataSet(data_dir = DATA_DIR, image_list_file = TRAIN_IMAGE_LIST, transform = data_transforms["train"])
val_dataset = ChestXrayDataSet(data_dir = DATA_DIR, image_list_file = VAL_IMAGE_LIST, transform = data_transforms["val"])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)
dataloaders = {"train": train_loader, "val": val_loader}

#image_name, image, classes = next(iter(train_loader))

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

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    fopen = open("accuracy.txt", "w")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_AUROC_avg = 0.0
    losses = Counter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print("Current Learning rate : {}".format(optimizer.param_groups[0]['lr']))
            gt = torch.FloatTensor().to(device)
            pred = torch.FloatTensor().to(device)
            losses.reset()
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            # Iterate over data.
            t = tqdm(enumerate(dataloaders[phase]),  desc='Loss: **** ', total=len(dataloaders[phase]), bar_format='{desc}{bar}{r_bar}')
            for batch_idx, (image_name, inputs, labels) in t:
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(inputs.shape, labels.shape)
               
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    gt = torch.cat((gt, labels), 0)
                    pred = torch.cat((pred, outputs.data), 0)
                    
                    #print(outputs.shape)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                losses.update(loss.data.item(), inputs.size(0))
                t.set_description('Loss: %.3f ' % (losses.avg))
            
            AUCs = compute_AUCs(gt, pred)
            AUROC_avg = np.array(AUCs).mean()
            
            if phase == 'train':
                scheduler.step()
                
            if phase == "val":
                if best_AUROC_avg < AUROC_avg:
                    best_AUROC_avg = AUROC_avg
                    #torch.save(model.state_dict(), "./checkpoint/fulldata_triplet_initialized0001.pth")
                fopen.write('\nEpoch {} \t [{}] : \t {AUROC_avg:.3f}\n'.format(epoch, phase, AUROC_avg=AUROC_avg))
                for i in range(N_CLASSES):
                    fopen.write('{} \t {}\n'.format(CLASS_NAMES[i], AUCs[i]))
                fopen.write('-' * 100)
                    
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, batch_idx + 1, len(dataloaders[phase]), loss=losses))
            print('{} : \t {AUROC_avg:.3f}'.format(phase, AUROC_avg=AUROC_avg))
            
            fopen.flush()
    fopen.close()
    return model


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

    
model_ft = model = ResidualNet('ImageNet', 50, 8, "TripletAttention")
print(model_ft)
#model_ft = initialize_model(model_ft)
print("Model Dictionary Initialized with pre-trained weights")

model_ft.load_state_dict(torch.load("./checkpoint/fulldata_triplet_initialized.pth"))
print("Model Weight Loaded")
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, N_CLASSES)
# #model_ft.load_state_dict(torch.load("./checkpoint/output14.pth"))

model_ft = model_ft.to(device)

criterion = nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)