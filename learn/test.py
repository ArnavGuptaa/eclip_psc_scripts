'''
Description: 

Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-05-02 10:39:19
'''
import torch
import pandas as pd
import pickle
import argparse
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import models
import numpy as np
from sklearn.metrics import accuracy_score
import data
from torch.utils.data import DataLoader
import os


def test(model, dataset):
    predict = []
    truth = []
    sums = 0
    with torch.no_grad():
        # model.eval()
        for _, j in enumerate(dataset):
            pos, neg = j
            # pos = pos # .cuda()
            # neg = neg.cuda()
            pre_pos, _ = model(pos)
            _, pre_neg = model(neg)
            #print(pre_pos)
            pre_pos = pre_pos.squeeze().cpu().numpy()
            #print(pre_pos.shape)

            pre_pos_binary = (pre_pos > 0.5).astype(int)
            sums += np.sum(pre_pos)

            # pre_neg = pre_neg.squeeze().cpu().numpy()
            # pre_neg_binary = (pre_neg > 0.5).astype(int)
            # print(pre_neg_binary)
            # print(pre_pos_binary)
            # predict.append(pre_neg_binary)
            # truth.append(np.zeros_like(pre_neg))
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
    res = accuracy_score(predict, truth)
    return res, sums / len(dataset)


parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    required=True,
                    type=str,
                    help="path to the model")
parser.add_argument("--save",
                    default="figures/",
                    type=str,
                    help="save path of the figure")
parser.add_argument("--testfasta", required=True, type=str, help="test fasta")
parser.add_argument("--inputlength",
                    default=80,
                    type=int,
                    help="the model input length")
parser.add_argument("--testinterval",
                    default=20,
                    type=int,
                    help="the stride of the window")
parser.add_argument("-ds", default=None, type=str, help="path to the dataset")
args = parser.parse_args()

name = os.path.basename(args.testfasta)
name = name[:name.find(".")]

if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.ds is not None:
    namet = os.path.basename(args.ds)
    namet = namet[:namet.find(".")]
    name = name + "   " + namet

model = torch.load(args.model, map_location=torch.device('cpu'))
model.eval()

if args.ds is not None:
    # ds = data.getbackdataset(args.ds, 40)
    ds = data.getdataset(args.ds, args.inputlength)
    dataset = DataLoader(ds, batch_size=50, shuffle=True)
    _, comp = test(model, dataset)
    # print(comp)
else:
    comp = None
print("finished loading dataset")

mapping = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}

# a = open("./COVID.fasta", "r")
a = open(args.testfasta, "r")
a.readline()
covid = []
for i in a.readlines():
    i = i.strip()
    covid.append(i)

wholeseq = "".join(covid)
wholeseq = [mapping[i] for i in list(wholeseq)]
print(len(wholeseq))
input = np.zeros((len(wholeseq) // args.testinterval - 10, args.inputlength))
for i in range(len(wholeseq) // args.testinterval - 10):
    input[i] = wholeseq[i * args.testinterval:i * args.testinterval +
                        args.inputlength]

input = torch.tensor(input)
print("finish loading test set")
# input = torch.randint(0, 4, (40, 40))
# print(input)
with torch.no_grad():
    # res, prob = model(input.long().cuda())
    res, prob = model(input.long())
    res = res.cpu().numpy()
    prob = prob.cpu().numpy()
    ran = torch.randint(0, 4, (10000, 80))
    # _, ran = model(ran.long().cuda())
    _, ran = model(ran.long())
    ran = torch.mean(ran).item()
    # print(ran, comp)
    if comp is None:
        res = -np.log(1 - prob + 1e-20)
    else:
        res = res / comp
        res[res < 0] = 0
        # print(np.mean(res))
        res = res * (0.5 / np.mean(res))
        res[res > 1] = 1
        res = np.exp(res) - 1
# print(np.min(res), np.max(res), ran)
plt.plot(res)
plt.title(name)
plt.savefig(os.path.join(args.save, name + ".png"))
plt.close()
