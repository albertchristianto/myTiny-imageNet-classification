# myTiny-imageNet-classification
This repository is a minimum implementation to train a image classification network on tiny ImageNet dataset. It is recommended for beginners in learning deep learning computer vision. It has detailed instruction to train on your custom dataset.

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
* You can also resume your training using this following command. The checkpoint file is in the checkpoint folder.
```
python3 train.py --resume --checkpoint_dir [YOUR_CHECKPOINT_PATH]
```
* To monitor the training loss, you can run this following command on another command window and open your browser to access the provided address
```
tensorboard --logdir [YOUR_CHECKPOINT_PATH]
```
## Test a Image Classification Network
To test a image classification, you can run this following command
```
python3 test.py --val_txtPath [PATH_TO_VALIDATION_TXT_FILE] --weight_path [PATH_TO_YOUR_PTH_FILE]
```

## Inference Code
To see the classification result of a image, run the following command
```
python3 demo.py ----img_Path [PATH_TO_A_IMAGE_FILE] --weight_path [PATH_TO_YOUR_PTH_FILE]
```

## Training on Custom Dataset
To train a image classification on your custom dataset, you can follow these steps:
* Set your dataset format look like this
```
${DATA_ROOT}
|-- YOUR_DATASET
    |-- train
        |-- your_class_0
            |-- 000000000009.jpg
            |-- ... 
        |-- your_class_1
            |-- 000000000139.jpg
            |-- ...
        ...
    |-- validation
        |-- your_class_0
            |-- 000000000009.jpg
            |-- ... 
        |-- your_class_1
            |-- 000000000139.jpg
            |-- ...
        ...

```
* Generate your train.txt, val.txt, and test.txt for your dataset using this format. For reference, you can see [here](https://github.com/albertchristianto/myTiny-imageNet-classification/blob/master/datalist/test.txt).
```
{DATA_ROOT}/YOUR_DATASET/train/your_class_0/000000000009.jpg 0
{DATA_ROOT}/YOUR_DATASET/train/your_class_1/000000000139.jpg 1
{IMG_PATH} {LABEL}
...
```
* Run this following command
```
python3 train.py --train_txtPath [PATH_TO_TRAIN_TXT_FILE] --val_txtPath [PATH_TO_VALIDATION_TXT_FILE]
```
* For advanced training, run this command to see the training option
```
python3 train.py -h
```
## License
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
