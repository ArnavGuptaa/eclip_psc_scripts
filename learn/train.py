'''
Description: 

Author: Tianyi Fei
Date: 2022-04-14 18:00:30
LastEditors: Tianyi Fei
LastEditTime: 2022-05-02 11:18:48
'''
'''
Description: 
Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-04-07 18:50:39
'''
import torch
import torch.nn as nn
import torch.optim as optim
# from timeit import default_timer as timer
import numpy as np
# from sklearn.metrics import accuracy_score
import data
import models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import argparse
import os
# from tqdm import tqdm
import copy


def test(model, dataset):
    predict = []
    truth = []
    # print(len(dataset))
    with torch.no_grad():
        model.eval()
        for _, j in enumerate(dataset):
            pos, neg = j
            # pos = pos.cuda()
            # neg = neg.cuda()
            _, pre_pos = model(pos)
            _, pre_neg = model(neg)

            pre_pos = pre_pos.squeeze().cpu().numpy()
            pre_pos_binary = (pre_pos > 0.5).astype(int)

            pre_neg = pre_neg.squeeze().cpu().numpy()
            pre_neg_binary = (pre_neg > 0.5).astype(int)
            # print(pre_neg_binary)
            # print(pre_pos_binary)
            predict.append(pre_neg_binary)
            truth.append(np.zeros_like(pre_neg))
            predict.append(pre_pos_binary)
            truth.append(np.ones_like(pre_pos))
            # for l in range(effi.shape[0]):
            #     for m in range(20):
            #         for k in range(20):
            #             if not nans[l][m][k]:
            #                 predict[m].append(pre_corr[l][m][k])
            #                 truth[m].append(corr[l][m][k])

    predict = np.concatenate(predict)
    truth = np.concatenate(truth)
    # print(predict)
    # print(pre_neg)
    # print(pre_pos)
    # print(predict)
    # print(truth)
    acc = np.sum(predict == truth)
    res = acc / len(predict)
    return res


def train(model, epoch, trainset, testset):
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    acc_test = []
    bestmodel = None
    best = -1
    for i in range(epoch):
        # print("Epoch ", i)
        total_loss = 0
        # start = timer()
        model.train()
        for _, j in enumerate(trainset):
            pos, neg = j
            # pos = pos.cuda()
            # neg = neg.cuda()
            # for k in range(4):
            #     print(sum(pos == k))
            #     print(sum(neg == k))
            # exit()
            _, pre_pos = model(pos)
            _, pre_neg = model(neg)

            pre_pos = pre_pos.squeeze()
            pre_neg = pre_neg.squeeze()

            loss = criterion(pre_pos, torch.ones_like(pre_pos)) + \
            criterion(pre_neg, torch.zeros_like(pre_neg))
            # loss = criterion(corr, pre_corr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(pre_pos)
        # print(pre_neg)
        # end = timer()
        # print("time elapsed ", end - start, "   loss: ", total_loss)
        res = test(model, testset)
        if res > best:
            res = best
            bestmodel = copy.deepcopy(model)
        # print("test res", res)
        # res2 = test(model, trainset)
        acc_test.append(res)

        # print("train res", res2)
    return bestmodel, acc_test


def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",
                        type=str,
                        default="./models/",
                        help="the path to the folder for saving models")
    parser.add_argument("-ds",
                        required=True,
                        type=str,
                        help="path to the dataset")
    parser.add_argument("--model",
                        default="full",
                        type=str,
                        help="which model to train")
    parser.add_argument("--inputlength",
                        default=80,
                        type=int,
                        help="the model input length")
    parser.add_argument("--epoches",
                        default=100,
                        type=int,
                        help="number of epoches to train")
    args = parser.parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    return args


if __name__ == "__main__":
    args = Args()
    ds = args.ds
    ds = data.getdataset(ds, args.inputlength)
    # ds = data.getbackdataset(ds, 40)
    print("dataset length", len(ds))
    name = os.path.basename(args.ds)
    name = name[:name.find(".")]
    # print(os.path.join(args.checkpoints, name + ".pth"))
    trainset, testset = random_split(
        ds, [len(ds) * 4 // 5, len(ds) - len(ds) * 4 // 5])
    testset = DataLoader(testset, batch_size=20, shuffle=True)
    trainset = DataLoader(trainset, batch_size=40, shuffle=True)

    # cnn = models.TwoLayer(8, 80).cuda()

    # cnn = models.TwoLayerONE(8, 80).cuda()

    # cnn = models.Motif(8, 80).cuda()
    if args.model == "motif":
        cnn = models.Motif(8, args.inputlength)  # .cuda()
    elif args.model == "full":
        cnn = models.TwoLayer(8, args.inputlength)
    else:
        raise NotImplementedError

    print("start training")
    cnn, res = train(cnn, args.epoches, trainset, testset)
    # print(res, os.path.join(args.checkpoints, name + ".pth"))
    print("finish")
    torch.save(cnn, os.path.join(args.save, name + args.model + ".pth"))
