
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter

import faiss

class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PrecisionRecallF1_Faiss(nn.Module):
    def __init__(self, numRetrieved):
        super().__init__()
        
        self.numRetrieved = numRetrieved

    def forward(self, train_f, train_y, test_f, test_y):
        
        mpre, _ = cal_mP_top_test(train_f, test_f, train_y, test_y, self.numRetrieved)

        
        return mpre, None


def oneAgainstAllFaiss(y_true, hash_binary, cls_num_ac, top_r):
    """
    loop over each image as query and evaluate its performance based on top_r selection
    hash_binary: each row is a binary code 
    https://github.com/facebookresearch/faiss/issues/459
    """
    precisions, recalls, dist = [], [], []
    numSample = hash_binary.shape[0]
    # index = faiss.IndexFlatL2(hash_binary.shape[1])
    # index.add(np.ascontiguousarray(hash_binary.astype(np.float32)))

    # D, I = index.search(np.ascontiguousarray(hash_binary.astype(np.float32)), top_r+1)

    # https://github.com/facebookresearch/faiss/wiki/Binary-indexes
    index = faiss.IndexBinaryFlat(hash_binary.shape[1] * 8)
    index.add(np.ascontiguousarray(hash_binary.astype(np.uint8)))
    D, I = index.search(np.ascontiguousarray(hash_binary.astype(np.uint8)), top_r+1)

    # check whether the query index is inside the retrieved, since the first place of I may not be the index of query
    I_re = []
    for i in range(numSample):
        tmp = I[i].tolist()
        if i in set(tmp):
            tmp.remove(i)
        else:
            tmp = tmp[:-1]
        I_re.append(tmp)
    I_re = np.asarray(I_re)
    indexes = I_re

    assert indexes.shape[1] == top_r

    y_true_cls_num = []

    for _, label in enumerate(y_true):
        y_true_cls_num.append(cls_num_ac[label])
    
    y_true_cls_num = np.asarray(y_true_cls_num) - 1

    indexes_f = indexes.flatten()
    knn_labels = y_true[indexes_f.tolist(),].reshape(indexes.shape)
    knn_labels_ind = (knn_labels == y_true[:, None]).astype(int)
    
    precisions = knn_labels_ind.sum(axis=1) / top_r
    recalls = knn_labels_ind.sum(axis=1) / y_true_cls_num

    # return np.mean(np.array(precisions)), np.mean(np.array(recalls)), dists
    return np.mean(precisions), np.mean(recalls), None


def cal_mP_top_test(train_codes, test_codes, train_labels, test_labels, top_r):
    """ 
    calculation mean precision of top-r retrieval result
    train_codes: hash codes of training data
    test_codes: hash codes of testing data
    train_labels: ground truth labels of training dataset
    test_labels: ground truth labels of testing dataset
    top_r: top r retrieval
    """
    trainLabelCounter = Counter(train_labels)
    numTrainSample = train_codes.shape[0]
    
    trainBinCodes = np.sign(train_codes)
    trainBinCodes = (trainBinCodes == 1)
    trainBinCodes = trainBinCodes.astype(int)
        
    testBinCodes = np.sign(test_codes)
    testBinCodes = (testBinCodes == 1)
    testBinCodes = testBinCodes.astype(int)
    
    index = faiss.IndexBinaryFlat(train_codes.shape[1] * 8)
    index.add(np.ascontiguousarray(trainBinCodes.astype(np.uint8)))
    D, I = index.search(np.ascontiguousarray(testBinCodes.astype(np.uint8)), top_r)
    
    y_true_cls_num = []
    for _, label in enumerate(test_labels):
        y_true_cls_num.append(trainLabelCounter[label])
    y_true_cls_num = np.asarray(y_true_cls_num)
    
    indexes = I
    indexes_f = indexes.flatten()
    knn_labels = train_labels[indexes_f.tolist(),].reshape(indexes.shape)
    knn_labels_ind = (knn_labels == test_labels[:, None]).astype(int)
    
    precisions = knn_labels_ind.sum(axis=1) / top_r
    recalls = knn_labels_ind.sum(axis=1) / y_true_cls_num

    return np.mean(precisions), np.mean(recalls)


class QuantizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, relaxedCodes):
        diff = torch.abs(relaxedCodes) - torch.ones(relaxedCodes.size()).cuda()
        qLoss = torch.mean(torch.abs(torch.pow(diff, 2)))
    
        return qLoss













