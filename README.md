# myTiny-imageNet-classification
This repository is a minimum implementation to train a image classification network on tiny ImageNet dataset. It is recommended for beginners in learning deep learning computer vision. It has a detailed explanation for each part of the code, which is used in image classification. It has detailed instruction to train on your custom dataset.

## Requirements
* Ubuntu 16.04 LTS
* Python 3.5
* PyTorch 0.4.1+
* Torchvision
* TensorboardX
* OpenCV
* Numpy

## Installation
In your command terminal, type these command below
* Installing git (skip this step if git has been installed)<br>
```
sudo apt-get install git
```
* Clone this repository and go to the repository directory<br>
```
git clone https://github.com/albertchristianto/myTiny-imageNet-classification
cd myTiny-imageNet-classification
```
* Install these python libraries <br>
```
sudo pip3 install opencv-python numpy tensorboardx
sudo pip3 install pytorch torchvision
```
* You are ready to run the code

## Train a Image Classification Network
To train a image classification network using this repository, you must:
* Download tiny-ImageNet dataset [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and extract it
* Generate train.txt, validation.txt, and test.txt inside datalist folder by running the following commands
```
python3 utils/createDataList.py --tinyImagenetPath [TINY_IMAGENET_FOLDER_PATH]
```
* And it is ready to train the network 
```
python3 train.py
```


## Future Works
* 
