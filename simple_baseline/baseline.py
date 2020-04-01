import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, lr_scheduler
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import math
import json
import time 

from utils.preprocess import create_bow, create_ans_dict
from utils.VQA import VQA
from tqdm import tqdm, trange
import jpeg4py 
from torchsummary import summary 
from albumentations.pytorch import ToTensor
from albumentations import (Compose, CenterCrop, VerticalFlip, RandomSizedCrop,
                            HorizontalFlip, HueSaturationValue, ShiftScaleRotate,
                            Resize, RandomCrop, Normalize, Rotate, Normalize)

writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(device)

# used constants
learning_rate = 0.002
num_epochs = 20
batch_size = 256 
h, w = 224, 224
thresh = 128

cnn = models.mobilenet_v2(pretrained=True)
cnn = nn.Sequential(*list(cnn.children())[:-2])
cnn = cnn.to(device)
# cnn.eval()
optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
lr_scheduler_cnn = torch.optim.lr_scheduler.StepLR(optimizer_cnn, step_size=2, gamma=0.5)


# summary(cnn, (3, h, w))

ans_dict = create_ans_dict('VQAv2/train/annotations/annotations.json') 
bag_of_words = create_bow('VQAv2/train/questions/questions.json', thresh)

train_dataset = VQA('VQAv2/train', bag_of_words=bag_of_words, ans_dict=ans_dict, mode='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                pin_memory=True, num_workers=8, drop_last=True) 

valid_dataset = VQA('VQAv2/val', bag_of_words=bag_of_words, ans_dict=ans_dict, mode='val')
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                pin_memory=True, num_workers=8, drop_last=True) 

img_embed_size = 1024
q_embed_size = train_dataset.get_q_embed_size()
n_ans = train_dataset.get_n_ans()

model = nn.Sequential(nn.Linear(img_embed_size + q_embed_size, n_ans)).to(device)

error = nn.CrossEntropyLoss(ignore_index = -1)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(epoch, print_every = 100):
    model.train()
    cnn.train() 
    
    n_samples = len(train_loader)
    start = time.time()
    print_loss = 0.0 
    correct = 0.0
    total = 0.0

    for i, (imgs, qs, anss) in enumerate(train_loader):
        imgs, qs, anss = imgs.to(device), qs.to(device), anss.to(device)

        optimizer.zero_grad()

        img_embeds = cnn(imgs).reshape((-1, img_embed_size))
        txt_embeds = qs
        inputs = torch.cat((img_embeds, txt_embeds), 1)
        
        outputs = model(inputs)

        loss = error(outputs, anss)

        loss.backward()

        optimizer.step()
        optimizer_cnn.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += anss.size(0)
        correct += (predicted == anss).sum().item()
        
        print_loss += loss.item()

        writer.add_scalar('Train/Loss', loss.item(), i + epoch * n_samples)
        writer.flush()

        if (i + 1) % print_every == 0:
            loss_avg = print_loss / print_every
            acc = correct / total
            correct = 0.0
            total = 0.0
            print_loss = 0.0
            print('%s (%d iters, %d%%)\tLoss: %.4f\tAccuracy: %.4f' % (timeSince(start, (i + 1) / len(train_loader)),
                                        (i + 1) * batch_size, (i + 1) / len(train_loader) * 100, loss_avg, 
                                        acc))
def evaluate(epoch):
    model.eval()
    cnn.eval() 
    
    loss = 0.0
    acc = 0.0 
    correct = 0.0
    total = 0.0
    
    with torch.no_grad():
        for i, (imgs, qs, anss) in enumerate(tqdm(valid_loader)):
            imgs, qs, anss = imgs.to(device), qs.to(device), anss.to(device)

            img_embeds = cnn(imgs).reshape((-1, img_embed_size))
            txt_embeds = qs
            inputs = torch.cat((img_embeds, txt_embeds), 1)
            
            outputs = model(inputs)

            loss += error(outputs, anss).item()

            _, predicted = torch.max(outputs.data, 1)
            total += anss.size(0)
            correct += (predicted == anss).sum().item()
    
    loss /= len(valid_loader)
    acc = correct / total
    
    print('Val loss: {:4f}'.format(loss))
    print('Val acc: {:4f}\n'.format(acc))

    writer.add_scalar('Val/Loss', loss, epoch)
    writer.add_scalar('Val/Accuracy', acc, epoch)
    writer.flush()

for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch + 1))
    train(epoch)
    evaluate(epoch)
    lr_scheduler.step()
    lr_scheduler_cnn.step()
