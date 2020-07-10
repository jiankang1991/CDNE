

import glob
from collections import defaultdict
import os
import numpy as np
import random


import torchvision.transforms as transforms

from PIL import Image
from skimage import io

def default_loader(path):
    return Image.open(path).convert('RGB')


def eurosat_loader(path):
    return io.imread(path)

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    return np.eye(n_classes)[x]

class DataGeneratorSplitting:
    """
    generate train and val dataset based on the following data structure:
    Data structure:
    └── SeaLake
        ├── SeaLake_1000.jpg
        ├── SeaLake_1001.jpg
        ├── SeaLake_1002.jpg
        ├── SeaLake_1003.jpg
        ├── SeaLake_1004.jpg
        ├── SeaLake_1005.jpg
        ├── SeaLake_1006.jpg
        ├── SeaLake_1007.jpg
        ├── SeaLake_1008.jpg
        ├── SeaLake_1009.jpg
        ├── SeaLake_100.jpg
        ├── SeaLake_1010.jpg
        ├── SeaLake_1011.jpg
    """

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.sceneFilesNum = defaultdict()
        
        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.CreateIdx2fileDict()


    def CreateIdx2fileDict(self):
        # import random
        # random.seed(42)

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        for label, scenePth in enumerate(self.sceneList):
            self.scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.seed(42)
            random.shuffle(subdirImgPth)

            # train_subdirImgPth = subdirImgPth[:int(0.2*len(subdirImgPth))]
            # val_subdirImgPth = subdirImgPth[int(0.2*len(subdirImgPth)):int(0.3*len(subdirImgPth))]
            # test_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):]
            
            train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
            val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)
                train_count += 1
            
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
            
            for imgPth in val_subdirImgPth:
                self.val_idx2fileDict[val_count] = (imgPth, label)
                val_count += 1
        
        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))

    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        elif self.phase == 'val':
            idx = self.valDataIndex[index]
        else:
            idx = self.testDataIndex[index]
        
        return self.__data_generation(idx)

            
    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]

        if self.phase == 'train':
            imgPth, imgLb = self.train_idx2fileDict[idx]
        elif self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        if self.dataset == 'eurosat':
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)
        oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb, 'idx':idx, 'onehot':oneHotVec.astype(np.float32)}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        # return {'img': img, 'label': imgLb}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)




















