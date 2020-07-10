""" 
memory bank for SNCA_CE_Hashing
"""

import os
import random
import math
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import shutil


import argparse
from tensorboardX import SummaryWriter


import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


import sys
sys.path.append('../')


from utils.LinearAverage import LinearAverage
from utils.NCA import NCACrossEntropy
from utils.model import ResNet18_cls
from utils.metrics import MetricTracker, PrecisionRecallF1_Faiss, QuantizationLoss
from utils.dataGen import DataGeneratorSplitting



parser = argparse.ArgumentParser(description='PyTorch SNCA Training for RS')
parser.add_argument('--data', metavar='DATA_DIR',  default='../data',
                        help='path to dataset (default: ../data)')
parser.add_argument('--dataset', metavar='DATASET',  default='ucmerced',
                        help='learning on the dataset (ucmerced)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='num_workers for data loading in pytorch, (default:8)')
parser.add_argument('--epochs', default=130, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--dim', default=128, type=int,
                    metavar='D', help='embedding dimension (default:128)')
parser.add_argument('--imgEXT', metavar='IMGEXT',  default='tif',
                        help='img extension of the dataset (default: tif)')
parser.add_argument('--temperature', default=0.05, type=float,
                    metavar='T', help='temperature parameter')
parser.add_argument('--memory-momentum', '--m-mementum', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('--margin', default=0.0, type=float,
                    help='classification margin')
parser.add_argument('--model', metavar='MODEL',  default='resnet18',
                        help='CNN model (resnet18 or 50)')
parser.add_argument('--lambda_', default=1, type=float,
                    metavar='LAMBDA', help='penalty parameter')


args = parser.parse_args()

sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
print('saving file name is ', sv_name)

checkpoint_dir = os.path.join('./', sv_name, 'checkpoints')
logs_dir = os.path.join('./', sv_name, 'logs')

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))

def save_checkpoint(state, is_best, name):

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth.tar')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name + '_model_best.pth.tar'))


def main():
    global args, sv_name, logs_dir, checkpoint_dir

    write_arguments_to_file(args, os.path.join('./', sv_name, sv_name+'_arguments.txt'))

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data_transform = transforms.Compose([
                                        transforms.Resize((256,256)),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])

    val_data_transform = transforms.Compose([
                                            transforms.Resize((256,256)),
                                            transforms.ToTensor(),
                                            normalize])

    train_dataGen = DataGeneratorSplitting(data=args.data, 
                                            dataset=args.dataset, 
                                            imgExt=args.imgEXT,
                                            imgTransform=train_data_transform,
                                            phase='train')
    
    train_dataGen_wo_shuf = DataGeneratorSplitting(data=args.data, 
                                            dataset=args.dataset, 
                                            imgExt=args.imgEXT,
                                            imgTransform=val_data_transform,
                                            phase='train')

    val_dataGen = DataGeneratorSplitting(data=args.data, 
                                            dataset=args.dataset, 
                                            imgExt=args.imgEXT,
                                            imgTransform=val_data_transform,
                                            phase='val')

    train_data_loader = DataLoader(train_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    trainloader_wo_shuf = DataLoader(train_dataGen_wo_shuf, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)


    model = ResNet18_cls(clsNum=len(train_dataGen.sceneList), dim=args.dim).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4, nesterov=True)

    best_acc = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            lemniscate = checkpoint['lemniscate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # define lemniscate and loss function (criterion)
        ndata = len(train_dataGen)
        lemniscate = LinearAverage(args.dim, ndata, args.temperature, args.memory_momentum).cuda()

    y_true = []

    for idx, data in enumerate(tqdm(trainloader_wo_shuf, desc="extracting training labels")):

        label_batch = data['label'].to(torch.device("cpu"))

        y_true += list(np.squeeze(label_batch.numpy()).astype(np.float32))
    
    y_true = np.asarray(y_true)

    CELoss = torch.nn.CrossEntropyLoss().cuda()

    QLoss = QuantizationLoss().cuda()

    criterion = NCACrossEntropy(torch.LongTensor(y_true),
            args.margin / args.temperature).cuda()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))

    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        adjust_memory_update_rate(lemniscate, epoch)

        train(train_data_loader, model, lemniscate, criterion, CELoss, QLoss, args.lambda_, optimizer, epoch, train_writer)
        acc, y_train = val(val_data_loader, trainloader_wo_shuf, model, epoch, val_writer)

        is_best_acc = acc > best_acc
        best_acc = max(best_acc, acc)
        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best_acc, sv_name)

        scheduler.step()


def train(trainloader, model, lemniscate, criterion, CELoss, QLoss, lambda_, optimizer, epoch, train_writer):

    losses = MetricTracker()
    sncalosses = MetricTracker()
    celosses = MetricTracker()
    quanlosses = MetricTracker()

    model.train()

    for idx, data in enumerate(tqdm(trainloader, desc="training")):

        imgs = data['img'].to(torch.device("cuda"))
        index = data["idx"].to(torch.device("cuda"))
        labels = data['label'].to(torch.device("cuda"))

        f, embedding, logits = model(imgs)
        output = lemniscate(embedding, index)
        
        sncaloss = criterion(output, index)
        celoss = CELoss(logits, labels)
        qloss = QLoss(f)

        loss = lambda_ * sncaloss + celoss + qloss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))
        sncalosses.update(sncaloss.item(), imgs.size(0))
        celosses.update(celoss.item(), imgs.size(0))
        quanlosses.update(qloss.item(), imgs.size(0))

    info = {
        "Loss": losses.avg,
        "SNCALoss": sncalosses.avg,
        "CELoss": celosses.avg,
        "QLoss": quanlosses.avg
    }
    for tag, value in info.items():
        train_writer.add_scalar(tag, value, epoch)

    print('Train TotalLoss: {:.6f} SNCALoss: {:.6f} CELoss: {:.6f} QLoss: {:.6f}'.format(
            losses.avg,
            sncalosses.avg,
            celosses.avg,
            quanlosses.avg
            ))


def val(valloader, trainloader_wo_shuf, model, epoch, val_writer):

    model.eval()
    
    metricHashingPRF1 = PrecisionRecallF1_Faiss(numRetrieved=20)

    train_features = []
    train_y_true = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(trainloader_wo_shuf, desc="extracting training data embeddings")):

            imgs = data['img'].to(torch.device("cuda"))
            label_batch = data['label'].to(torch.device("cpu"))

            _, embedding, _ = model(imgs)

            train_features += list(embedding.cpu().numpy().astype(np.float32))
            train_y_true += list(np.squeeze(label_batch.numpy()).astype(np.float32))
    
    train_features = np.asarray(train_features)
    train_y_true = np.asarray(train_y_true)

    y_val_true = []
    val_features = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valloader, desc="validation")):

            imgs = data['img'].to(torch.device("cuda"))
            label_batch = data['label'].to(torch.device("cpu"))

            _, embedding, _ = model(imgs)

            val_features += list(embedding.cpu().numpy().astype(np.float32))
            y_val_true += list(np.squeeze(label_batch.numpy()).astype(np.float32))

    y_val_true = np.asarray(y_val_true)
    val_features = np.asarray(val_features)

    mpre, _ = metricHashingPRF1(train_features, train_y_true, val_features, y_val_true)

    info = {
            'HashingPrec':mpre,
            # 'HammingBallRadiusPrec':hammingBallRadiusPrec.val
            # 'HashingRec':hashingRec.val,
            # 'HashingF1':meters["HashingF1"].val
        }
    for tag, value in info.items():
        val_writer.add_scalar(tag, value, epoch)

    print('Validation HashingPrec: {:.6f} '.format(
            mpre,
            # hammingBallRadiusPrec.val,
            ))
    
    return mpre, train_y_true



def adjust_memory_update_rate(lemniscate, epoch):
    if epoch >= 80:
        lemniscate.params[1] = 0.8
    if epoch >= 120:
        lemniscate.params[1] = 0.9


if __name__ == "__main__":
    main()
