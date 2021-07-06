from __future__ import print_function, division
from tqdm import tqdm
from read_data import ChestXrayDataSet, augument
from sklearn.metrics import roc_auc_score
from augmentation import ColourDistortion, BlurOrSharpen
from critic import LinearCritic
import torch.nn.functional as F

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
import pickle
from c_resnet import *
from utils import *

TRAIN_IMAGE_LIST = "train.txt"
VAL_IMAGE_LIST = "test.txt"
DATA_DIR = "/data/NIH_Xray/images/"
with open('patient2image.pickle', 'rb') as fp:
    patient2image = pickle.load(fp)
with open('image2patient.pickle', 'rb') as fp:
    image2patient = pickle.load(fp)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'c_train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

num_epochs, batch_size, workers, train_sampler, N_CLASSES = 20, 24, 4, None, 8
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax']

train_dataset = ChestXrayDataSet(data_dir = DATA_DIR, image_list_file = TRAIN_IMAGE_LIST, transform = data_transforms["train"])
val_dataset = ChestXrayDataSet(data_dir = DATA_DIR, image_list_file = VAL_IMAGE_LIST, transform = data_transforms["val"])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)
dataloaders = {"train": train_loader, "val": val_loader}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
def train_model(model, criterion, optimizer, scheduler, critic, c_criterion, c_optimizer, c_lr_scheduler, epoch):
    print("Training Epoch -> {}".format(epoch))
    print('-' * 100)
    losses = Counter()
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    model.train()
    critic.train()
    alpha = 0.2
    t = tqdm(enumerate(dataloaders["train"]),  desc='Loss: **** ', total=len(dataloaders["train"]), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (image_name, inputs, labels) in t:
        aug1, aug2 = augument(image_name, data_transforms["c_train"], patient2image, image2patient)
        aug1, aug2 = aug1.to(device), aug2.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        
        optimizer.zero_grad()
        c_optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs, _ = model(inputs)
            gt = torch.cat((gt, labels), 0)
            pred = torch.cat((pred, outputs.data), 0)
            loss = criterion(outputs, labels)
            
            _, representation1 = model(aug1) 
            _, representation2 = model(aug2)
            raw_scores, pseudotargets = critic(representation1, representation2)
            c_loss = c_criterion(raw_scores, pseudotargets)
            
            total_loss = (alpha * c_loss) + ((1 - alpha) * loss)
            total_loss.backward()
            
            optimizer.step()
            c_optimizer.step()
            
        losses.update(total_loss.data.item(), inputs.size(0))
        t.set_description('Loss: %.3f ' % (losses.avg))
    AUCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUCs).mean()
    scheduler.step()
    print('\nEpoch {} \t [{}] : \t {AUROC_avg:.3f}\n'.format(epoch, "Train ", AUROC_avg=AUROC_avg))
    return model, critic

def eval_model(model, fopen, epoch):
    losses = Counter()
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    losses.reset()

    model.eval()   # Set model to evaluate mode
    with torch.no_grad():
        t = tqdm(enumerate(dataloaders["val"]),  desc='Loss: **** ', total=len(dataloaders["val"]), bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (image_name, inputs, labels) in t:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs)
            gt = torch.cat((gt, labels), 0)
            pred = torch.cat((pred, outputs.data), 0)
        AUCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUCs).mean()
    print('\nEpoch {} \t [{}] : \t {AUROC_avg:.3f}\n'.format(epoch, "Validation", AUROC_avg=AUROC_avg))
    fopen.write('\nEpoch {} \t [{}] : \t {AUROC_avg:.3f}\n'.format(epoch, "Validation", AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        fopen.write('{} \t {}\n'.format(CLASS_NAMES[i], AUCs[i]))
    return AUROC_avg

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
        
        
def main():
    fopen = open("accuracy_BCE_85.txt", "w")
    model_ft = model = ResidualNet('ImageNet', 50, 8, "TripletAttention")
    initialize_model(model_ft)
    #model_ft.load_state_dict(torch.load("./checkpoint/model_BCE_837.pth"))
    print("Model created and pre-trained weights loaded ...")
    model_ft = model_ft.to(device)
    
    critic = LinearCritic(2048, temperature=0.5).to(device)
    c_criterion = nn.CrossEntropyLoss()
    c_optimizer = optim.SGD(list(model_ft.parameters()) + list(critic.parameters()), lr=0.000001, weight_decay=1e-6,
                           momentum=0.9)

    c_lr_scheduler = lr_scheduler.StepLR(c_optimizer, step_size=7, gamma=0.1)
    #encoder_optimizer = LARS(c_optimizer, trust_coef=1e-3)
    
    criterion = nn.BCEWithLogitsLoss()
    #criterion = FocalLoss(logits = True)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    best_auc = 0.0
    for epoch in range(1, num_epochs + 1):
        model_ft, critic = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, critic, c_criterion, c_optimizer, c_lr_scheduler, epoch)
        auc = eval_model(model_ft, fopen, epoch)
        if auc > best_auc:
            best_auc = auc
            torch.save(model_ft.state_dict(), "./checkpoint/model_BCE85.pth")
            torch.save(critic.state_dict(), "./checkpoint/critic_BCE85.pth")
        fopen.flush()
    
if __name__ == '__main__':
    main()
