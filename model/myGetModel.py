import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import models
from myVGG16 import myVGG16
from myAlexnet import myAlexnet
from myOwnModel import myOwnModel
from myResNet18 import myResNet18

def getModels(inp_size, trunk='myVGG16', numClass = 0, pretrained_choice=False):
	'''this program is used to build CNN model with input size inp_sizexinp_sizex3'''
	if trunk == 'myVGG16' and numClass!= 0:
		myModel = myVGG16(numClass,inp_size)

		if pretrained_choice:
			vgg16_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
			model_path = '../input/pretrained_resnet50'
			load_until = 20
			vgg_state_dict = model_zoo.load_url(vgg16_url, model_dir=model_path)

			weights_load = {}
			vgg_keys = vgg_state_dict.keys()
			for i in range(load_until):
				weights_load[list(myModel.state_dict().keys())[i]] = vgg_state_dict[list(vgg_keys)[i]]
			state = myModel.state_dict()
			state.update(weights_load)
			myModel.load_state_dict(state)
	elif trunk == 'myAlexnet' and numClass!= 0:
		myModel = myAlexnet(numClass,inp_size)
	elif trunk == 'mine' and numClass!= 0:
		myModel = myOwnModel(numClass,inp_size)
	elif trunk == 'myResNet18' and numClass!= 0:
		myModel = myResNet18(inp_size, pretrained=pretrained_choice, num_classes=numClass)
	else:
		print('Please define the model or number of class that you are using now')
		exit()
	return myModel
