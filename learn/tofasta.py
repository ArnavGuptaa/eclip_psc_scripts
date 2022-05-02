'''
Description: 

Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-05-02 14:27:48
'''
import argparse
import os
import pandas as pd
import numpy as np


def change(df, t):
    start = np.array(df[1])
    end = np.array(df[2])
    mid = (start + end) // 2
    df[1] = mid - t
    df[2] = mid + t
    return df


parser = argparse.ArgumentParser()
parser.add_argument("--length",
                    type=int,
                    default=80,
                    help="the length of input(will be rounded to even)")
parser.add_argument("--bed",
                    type=str,
                    required=True,
                    help="the input bed file")
args = parser.parse_args()

df = pd.read_csv(args.bed, header=None, sep="\t")
df = change(df, args.length // 2)
df.to_csv("./temp.bed", sep="\t", header=False, index=False)