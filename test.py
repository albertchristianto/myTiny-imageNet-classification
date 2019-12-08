import sys
import os
sys.path.append(os.getcwd())
sys.path.append('model')
sys.path.append('utils')
import argparse
import time

import torch
import torch.nn as nn

from model.myGetModel import getModels
from utils.myDataLoader import getLoader

def run():
	parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Classification Testing Code by Albert Christianto')
	parser.add_argument('--dontUseCUDA', action='store_false', default = True) 
	parser.add_argument('--val_txtPath', default='datalist/val.txt', type=str, metavar='DIR',
	                                        help='path to validation list')
	parser.add_argument('--num_class', default=200, type=int, metavar='N',
	                                        help='number of classes for this implementation')
	parser.add_argument('-b', '--batch_size', default=2, type=int,
	                                        metavar='N', help='batch size (default: 16)')
	parser.add_argument('--cpu_workers', default=0, type=int, metavar='N',
	                                        help='number cpu to run the program')
	parser.add_argument('--weight_path', default='checkpoint/myVGG16_best.pth', type=str, metavar='DIR',
	                                        help='path to weight of the model')   
	parser.add_argument('--model_type', type=str, default='myVGG16', help='define the model type that will be used: myVGG16, myAlexnet')
	parser.add_argument('--input_size', default=64, type=int, metavar='N',
	                                        help='number of epochs to save the model')
	args = parser.parse_args()

	#this is the setting for data augmentation
	transform = {}
	transform['random_horizontal_flips'] = 0.5
	transform['random_crop'] = 0.7
	transform['max_shifting'] = 4
	transform['input_size'] = args.input_size

	#BUILDING THE NETWORK
	print('Building {} network'.format(args.model_type))
	cnn_model = getModels(args.input_size,trunk=args.model_type, numClass = args.num_class)
	print('Finish building the network')
	print(cnn_model)

	#LOADING THE DATASET
	##validation
	valLoader = getLoader(args.val_txtPath, transform = transform, bsize = args.batch_size, nworkers = 0, dataShuffle = True)
	valDatasetSize = len(valLoader.dataset)
	print('validation dataset len: {}'.format(valDatasetSize))
	
	#load the trained network
	cnn_model.load_state_dict(torch.load(args.weight_path))

	#load the model and the criterion in the GPU
	if args.dontUseCUDA:
	    cnn_model.cuda()

	##Last validation------------------------------------------------------------------------- 
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
	print('Accuracy: {:.4f}'.format(val_acc))

if __name__ == '__main__':
	run()
