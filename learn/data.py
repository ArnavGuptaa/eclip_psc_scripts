'''
Description: 

Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-04-20 17:08:48
'''
import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import random
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import random_split
import pickle
import numpy as np
from torch.utils.data import DataLoader

mapping = {
    "A": 0,
    "T": 1,
    "G": 2,
    "C": 3,
    "N": 4,
    "n": 4,
    "M": 4,
    "R": 4,
    "Y": 4,
    "W": 4,
    "K": 4,
    "B": 4,
    "S": 4,
    "c": 3,
    "t": 1,
    "a": 0,
    "g": 2,
}


class SeqDataset(Dataset):
    def __init__(self, sequences, length):
        self.length = length
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        neg = torch.randint(0, 4, (self.length, ))
        pos = self.sequences[index]
        return pos, neg


class BackDataset(Dataset):
    def __init__(self, sequences, length, background):
        self.length = length
        self.sequences = sequences
        self.background = torch.tensor(background)
        self.backlength = len(background)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        start = random.randint(100, self.backlength - 100)
        neg = self.background[start:start + 40]
        pos = self.sequences[index]
        return pos, neg


def getdataset(path, length):
    seqs = []
    for record in SeqIO.parse(path, "fasta"):
        seq = record.seq
        seqs.append(torch.tensor([mapping[i] for i in list(seq)]))

    return SeqDataset(seqs, length)


def getbackdataset(path, length):
    seqs = []
    for record in SeqIO.parse(path, "fasta"):
        seq = record.seq
        seqs.append(torch.tensor([mapping[i] for i in list(seq)]))
    back = []
    for record in SeqIO.parse("/media/alvin/Elements/genomes/hg38.fa",
                              "fasta"):
        seq = record.seq
        back.append(str(seq))
    back = "".join(back)
    back = [mapping[i] for i in list(back)]
    return BackDataset(seqs, length, back)


if __name__ == "__main__":
    # ds = getProportionDataset("./ABEnew_proportion_dict.pkl")
    print(mapping.transform(list("AGCT")))