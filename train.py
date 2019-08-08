import os
import argparse
import time
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from model.myGetModel import getModels
from utils.myDataLoader import getLoader

#this is the setting for data augmentation
transform = {}
transform['random_horizontal_flips'] = 0.5
transform['random_crop'] = 0.7
transform['max_shifting'] = 4

parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Classification Training Code by Albert Christianto')
parser.add_argument('--dontUseCUDA', action='store_false', default = True) 
parser.add_argument('--train_txtPath', default='datalist/new_train.txt', type=str, metavar='DIR',
                    help='path to train list') 
parser.add_argument('--val_txtPath', default='datalist/new_val.txt', type=str, metavar='DIR',
                    help='path to validation list')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_class', default=200, type=int, metavar='N',
                    help='number of classes for this implementation')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--notNesterov', action='store_false',default = True) 
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='batch size (default: 256)')
parser.add_argument('--cpu_workers', default=4, type=int, metavar='N',
                    help='number cpu to run the program')
parser.add_argument('--checkpoint_dir', default='checkpoint', type=str, metavar='DIR',
                    help='path to save tensorboard log and weight of the model')   
parser.add_argument('--resume', action='store_true', default = False)
parser.add_argument('--pretrained', action='store_true', default = False)
parser.add_argument('--model_type', type=str, default='myVGG16', help='define the model type that will be used: myVGG16, myAlexnet')
parser.add_argument('--save_freq', default=2, type=int, metavar='N',
                    help='number of epochs to save the model')                                     
args = parser.parse_args()

#BUILDING THE NETWORK
print('Building {} network'.format(args.model_type))
cnn_model = getModels(args.model_type, numClass = args.num_class, pretrained_choice=args.pretrained)
print('Finish building the network')
#LOADING THE DATASET
##training
trainLoader = getLoader(args.train_txtPath, transform = transform, bsize = args.batch_size, nworkers = args.cpu_workers, dataShuffle = True)
trainDatasetSize = len(trainLoader.dataset)
print('train dataset len: {}'.format(trainDatasetSize))
##validation
valLoader = getLoader(args.val_txtPath, transform = transform, bsize = args.batch_size, nworkers = 2, dataShuffle = True)
valDatasetSize = len(valLoader.dataset)
print('validation dataset len: {}'.format(valDatasetSize))
#build loss criterion
criterion = nn.CrossEntropyLoss()
#build training optimizer
optimizer_cnn_model = optim.SGD(cnn_model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.notNesterov)
#build learning scheduler
lr_train_scheduler = lr_scheduler.StepLR(optimizer_cnn_model, step_size=10, gamma=0.1)
#load the model and the criterion in the GPU
if args.dontUseCUDA:
    cnn_model.cuda()
    criterion.cuda()

if args.resume:
    checkpoints_dir = args.checkpoint_dir
    train_checkpoints_path = os.path.join(checkpoints_dir,'training_checkpoint.pth.tar')
    checkpoint = torch.load(train_checkpoints_path)
    start_epoch = checkpoint['epoch']
    n_iter = checkpoint['n_iter']
    cnn_model.load_state_dict(checkpoint['model_state_dict'])
    lr_train_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    optimizer_cnn_model.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = os.path.join(args.checkpoint_dir,current_time)
    try:
        os.makedirs(checkpoints_dir)
    except os.error:
        pass
    train_checkpoints_path = os.path.join(checkpoints_dir,'training_checkpoint.pth.tar')
    start_epoch = 0
    n_iter = 0
#create tensorboard logging file 
writer = SummaryWriter(log_dir=checkpoints_dir)
#enable all cnn model training parameter
for param in cnn_model.parameters():
    param.requires_grad = True
#set cnn_model on the train mode
cnn_model.train()
#training process
for epoch in range(start_epoch, args.epochs):

    # train the network
    running_loss = 0.0
    running_corrects = 0
    for i, (img, label) in enumerate(trainLoader):
    	#load all the data in GPU
        if args.dontUseCUDA:
            img = img.cuda()
            label = label.cuda()
        #change the data type
        img = torch.autograd.Variable(img)
        label = torch.autograd.Variable(label)
        #set gradient to zero
    	optimizer_cnn_model.zero_grad()
        #inference the input
    	outputs = cnn_model(img)
        #get the training prediction
       	_, preds = torch.max(outputs, 1)
        #compute the loss
        loss = criterion(outputs, label)
        #compute the gradient
        loss.backward()
        #update the model
        optimizer_cnn_model.step()
        running_loss += loss.item() * img.size(0)
        running_corrects += torch.sum(preds == label.data)
        writer.add_scalar('Loss_Logging/loss_iteration',loss.item(),n_iter)
        n_iter += 1
        if i%50==0:
            print('iteration:{}. Loss:{}'.format(i,loss.item()))
    epoch_loss = running_loss / trainDatasetSize
    epoch_acc = running_corrects.double() / trainDatasetSize
    print('Epoch:{}. Loss: {:.4f}. Training Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
    writer.add_scalar('Loss_Logging/loss_epoch',epoch_loss,epoch)    
    writer.add_scalar('Accuracy/train_set',epoch_acc,epoch)    

    #save checkpoint, then validate the network
    if epoch % 5 == 0:
        #save checkpoint
        print('Saving checkpoint...')
        torch.save({'epoch':epoch,
                'n_iter':n_iter,
                'model_state_dict':cnn_model.state_dict(),
                'optimizer_state_dict':optimizer_cnn_model.state_dict(),
                'lr_scheduler_state_dict':lr_train_scheduler.state_dict()},train_checkpoints_path)
        #validate
        #set cnn_model on the val mode
        print('Validating...')
        cnn_model.eval()
        correct = 0
        for i, (img, label) in enumerate(valLoader):
            if args.dontUseCUDA:
                img = img.cuda()
                label = label.cuda()
            img = torch.autograd.Variable(img)
            label = torch.autograd.Variable(label)
            outputs = cnn_model(img)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == label.data)
        val_acc = correct.double() / valDatasetSize
        print('Validation accuracy: {:.4f}'.format(val_acc))
        writer.add_scalar('Accuracy/val_set',val_acc,epoch)
        #set cnn_model on the train mode
        cnn_model.train()

    #save the weight of the model
    if epoch%args.save_freq==0:
        model_save_filename=os.path.join(checkpoints_dir,'epoch_{}.pth'.format(epoch))
        torch.save(cnn_model.state_dict(),model_save_filename)

##Last validation------------------------------------------------------------------------- 
#save checkpoint
print('Saving checkpoint...')
torch.save({'epoch':epoch,
        'n_iter':n_iter,
        'model_state_dict':cnn_model.state_dict(),
        'optimizer_state_dict':optimizer_cnn_model.state_dict(),
        'lr_scheduler_state_dict':lr_train_scheduler.state_dict()},train_checkpoints_path)
#validate
#set cnn_model on the val mode
print('Validating...')
cnn_model.eval()
correct = 0
for i, (img, label) in enumerate(valLoader):
    if args.dontUseCUDA:
        img = img.cuda()
        label = label.cuda()
    img = torch.autograd.Variable(img)
    label = torch.autograd.Variable(label)
    outputs = cnn_model(img)
    _, preds = torch.max(outputs, 1)
    correct += torch.sum(preds == label.data)
val_acc = correct.double() / valDatasetSize
print('Validation accuracy: {:.4f}'.format(val_acc))
writer.add_scalar('Accuracy/val_set',val_acc,epoch)