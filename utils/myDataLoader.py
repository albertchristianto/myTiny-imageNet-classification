import cv2
import numpy as np
import random
import torch

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.numSample = len(data)
        self.transformation = transform

    def myPreprocess(self, image):
        image = image.astype(np.float32) / 255.
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = image.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

        preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
        return preprocessed_img

    def myAugmentation(self, img):
        if (random.random()>self.transformation['random_crop']):
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        else:
            dice_x = random.random()
            dice_y = random.random()

            x_offset = int((dice_x - 0.5) * 2 * self.transformation['max_shifting'])
            y_offset = int((dice_y - 0.5) * 2 * self.transformation['max_shifting'])

            center = np.array([32, 32])+ np.array([x_offset, y_offset])#center object became reference for cropping image
            center = center.astype(int)

            # pad up and down
            pad_v = np.ones((64, img.shape[1], 3), dtype=np.uint8) * 128
            
            img = np.concatenate((pad_v, img, pad_v), axis=0)
            
            # pad right and left
            pad_h = np.ones((img.shape[0], 64, 3), dtype=np.uint8) * 128
            
            img = np.concatenate((pad_h, img, pad_h), axis=1)
            
            img = img[int(center[1] + 64 / 2):int(center[1] + 64 / 2 + 64),
                      int(center[0] + 64 / 2):int(center[0] + 64 / 2 + 64), :]

        if (random.random()>self.transformation['random_horizontal_flips']):
            img = cv2.flip(img,1)
        return img

    def __getitem__(self, index):
        imgPath = self.data[index][0]
        label = int(self.data[index][1])
        img = cv2.imread(imgPath)
        img = self.myAugmentation(img)
        img = self.myPreprocess(img)
        img = torch.from_numpy(img)

        return img, label

    def __len__(self):
        return self.numSample

def getLoader(txtPath, transform, bsize = 16, nworkers = 8, dataShuffle = True):
    """ Build a dataloader
    :param txtPath: string, path to txt file
    :returns : the data loader
    """
    #load the data first
    datasetFile = []
    theFile = open(txtPath)
    theFile = theFile.readlines()
    for i, theFileNow in zip(range(len(theFile)), theFile):
        theFileNow =theFileNow.replace('\n','').split(' ')
        datasetFile.append(theFileNow)
    
    #load the dataset
    dataset = MyDataset(data=datasetFile, transform = transform)
    data_loader = DataLoader(dataset, batch_size = bsize, num_workers = 1, shuffle = True)

    return data_loader
