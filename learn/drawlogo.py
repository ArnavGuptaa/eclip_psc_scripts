'''
Description: 

Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-05-02 21:02:10
'''
import torch
import data
import models
import logomaker as lm
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--save",
                    type=str,
                    default="./figures/",
                    help="the path to the folder for saving logos")
parser.add_argument("--model",
                    required=True,
                    type=str,
                    help="path to the model")

args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)
model = torch.load(args.model)
for i in (model.conv1.parameters()):
    j = i
    break
j = j.detach().numpy()
j = j.swapaxes(1, 2)
j = j[:, :, :4]
name = os.path.basename(args.model)
name = name[:name.find(".")]
for i in range(16):
    df = pd.DataFrame(j[i])
    df.columns = ["A", "T", "G", "C"]
    logo = lm.Logo(df)
    plt.savefig(os.path.join(args.save, name + str(i) + ".png"))
    plt.close()
