import sys
import os
sys.path.append(os.getcwd())
sys.path.append('model')
sys.path.append('utils')
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from model.myGetModel import getModels

def myPreprocess(image):
	image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
	image = image.astype(np.float32) / 255.
	means = [0.485, 0.456, 0.406]
	stds = [0.229, 0.224, 0.225]

	preprocessed_img = image.copy()[:, :, ::-1]
	for i in range(3):
	    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
	    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

	preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
	preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
	preprocessed_img = torch.from_numpy(preprocessed_img)
	return preprocessed_img


def run():
	parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Classification Demo Code by Albert Christianto')
	parser.add_argument('--dontUseCUDA', action='store_false', default = True) 
	parser.add_argument('--img_Path', default='samples/Class-0.JPEG', type=str, metavar='DIR',
	                                        help='path to validation list')
	parser.add_argument('--num_class', default=200, type=int, metavar='N',
	                                        help='number of classes for this implementation')
	parser.add_argument('--cpu_workers', default=0, type=int, metavar='N',
	                                        help='number cpu to run the program')
	parser.add_argument('--weight_path', default='checkpoint/myVGG16_best.pth', type=str, metavar='DIR',
	                                        help='path to weight of the model')   
	parser.add_argument('--model_type', type=str, default='myVGG16', help='define the model type that will be used: myVGG16, myAlexnet')
	parser.add_argument('--input_size', default=64, type=int, metavar='N',
	                                        help='number of epochs to save the model')
	args = parser.parse_args()

	#BUILDING THE NETWORK
	print('Building {} network'.format(args.model_type))
	cnn_model = getModels(args.input_size,trunk=args.model_type, numClass = args.num_class)
	print('Finish building the network')
	print(cnn_model)

	#load the trained network
	cnn_model.load_state_dict(torch.load(args.weight_path))

	#load the model and the criterion in the GPU
	if args.dontUseCUDA:
	    cnn_model.cuda()

	#set cnn_model on the val mode
	cnn_model.eval()
	img = cv2.imread(args.img_Path)
	img = myPreprocess(img)
	if args.dontUseCUDA:
	    img = img.cuda()
	img = torch.autograd.Variable(img)
	output = cnn_model(img)
	_, pred = torch.max(output, 1)
	classNUM = pred.data.cpu().numpy()[0]
	print('Predicted as Class Index {}'.format(int(classNUM)))

if __name__ == '__main__':
	run()
