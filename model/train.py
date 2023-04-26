import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from model import ARSC_NET
from dataloader import  get_mnist_loaders


parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=1500)
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--data_aug', type=eval, default=False, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--save', type=str, default='log/')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--input', '-i', default='../analysis/pack/melspec_mfccs_train.p', type=str,
        help='path to directory with input data archives')
parser.add_argument('--test', default='../analysis/pack/melspec_mfccs_test.p', type=str,
        help='path to directory with test data archives')
args = parser.parse_args()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def confusion_matrix(model, dataset_loader):
    targets = []
    outputs = []
    total_loss = 0
    for mfccs,melspec, y in dataset_loader:
        mfccs = mfccs.to(device)
        melspec = melspec.to(device)
        y_ = y.to(device)
        y = one_hot(np.array(y.numpy()), 4)
        target_class = np.argmax(y, axis=1)
        targets = np.append(targets,target_class)
        logits = model(mfccs,melspec)
        predicted_class = np.argmax(logits.cpu().detach().numpy(), axis=1)
        outputs = np.append(outputs,predicted_class)
        
        entroy = nn.CrossEntropyLoss().to(device)
        loss = entroy(logits, y_).cpu().numpy()
        total_loss += loss

    Confusion_matrix=sk_confusion_matrix(targets.tolist(), outputs.tolist())
    pos_t=Confusion_matrix[0][0]
    neg_t=Confusion_matrix[1][1]+Confusion_matrix[2][2]+Confusion_matrix[3][3]
    pos_sum=Confusion_matrix[0].sum()
    neg_sum=Confusion_matrix[1:].sum()

    se = neg_t/neg_sum
    sp = pos_t/pos_sum
    icbhi_score = (se+sp)/2
    acc = (pos_t+neg_t)/(pos_sum+neg_sum)
    return Confusion_matrix,se,sp,icbhi_score,acc,total_loss  / (len(dataset_loader.dataset)/args.batch_size)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    model = ARSC_NET().to(device)
    
    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.batch_size
    )


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    epoch=0
    best_socre = 0
    for epoch in range(args.nepochs):
        logger.info(f"epoch:{format(epoch+1)}")
        optimizer.zero_grad()
        total_loss = 0
        targets=[]
        outputs=[]
        for (mfccs,melspec,y) in train_loader:
            mfccs = mfccs.to(device)
            melspec = melspec.to(device)
            y_ = one_hot(np.array(y.numpy()), 4)

            target_class = np.argmax(y_, axis=1)
            targets = np.append(targets,target_class)

            y = y.to(device)
            logits = model(mfccs, melspec)

            predicted_class = np.argmax(logits.cpu().detach().numpy(), axis=1)
            outputs = np.append(outputs,predicted_class)

            loss = criterion(logits, y)
            lossdata=loss.cpu().detach().numpy()
            total_loss += lossdata

            loss.backward()
            optimizer.step()

        epoch+=1

        Confusion_matrix=sk_confusion_matrix(targets.tolist(), outputs.tolist())
        pos_t=Confusion_matrix[0][0]
        neg_t=Confusion_matrix[1][1]+Confusion_matrix[2][2]+Confusion_matrix[3][3]
        pos_sum=Confusion_matrix[0].sum()
        neg_sum=Confusion_matrix[1:].sum()

        train_se = neg_t/neg_sum
        train_sp = pos_t/pos_sum
        train_icbhi_score = (train_se+train_sp)/2
        train_acc = (pos_t+neg_t)/(pos_sum+neg_sum)
        
        train_loss = total_loss  / (len(train_loader.dataset)/args.batch_size)
        logger.info(f"Train loss : {format(train_loss, '.4f')}\tTrain SE : {format(train_se, '.4f')}\tTrain SP : {format(train_sp, '.4f')}\tTrain Score : {format(train_icbhi_score, '.4f')}\tTrain Acc : {format(train_acc, '.4f')}")
        
        with torch.no_grad():
            Confusion_matrix,val_se, val_sp, val_icbhi_score, val_acc, val_loss=confusion_matrix(model, test_loader)
            logger.info(f"Val loss : {format(val_loss, '.4f')}\tVal SE : {format(val_se, '.4f')}\tVal SP : {format(val_sp, '.4f')}\tVal Score : {format(val_icbhi_score, '.4f')}\tVal Acc : {format(val_acc, '.4f')}") 
            
            if val_icbhi_score > best_socre:
                torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                best_socre = val_icbhi_score
                best_matrix=Confusion_matrix
    
        logger.info(f"best icbhi score is {format(best_socre, '.4f')}")   
        logger.info(best_matrix)