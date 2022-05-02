'''
Description: 

Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-05-01 23:44:02
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayer(nn.Module):
    def __init__(self, d_embed, length) -> None:
        super().__init__()
        self.embed = nn.Embedding(5, d_embed)
        self.conv1 = nn.Conv1d(d_embed, d_embed * 2, 3, padding="same")
        # self.bn1 = nn.BatchNorm1d(d_embed * 2)
        self.conv2 = nn.Conv1d(d_embed * 2, d_embed * 4, 3, padding="same")
        # self.bn2 = nn.BatchNorm1d(d_embed * 4)
        self.conv3 = nn.Conv1d(d_embed * 4, d_embed, 3, padding="same")
        # self.bn3 = nn.BatchNorm1d(d_embed)
        self.fc1 = nn.Linear(length * d_embed, length * d_embed // 2)
        self.fc2 = nn.Linear(length * d_embed // 2, 1)

    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(1, 2)
        # x = torch.relu(self.bn1(self.conv1(x)))
        # x = torch.relu(self.bn2(self.conv2(x)))
        # x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x, torch.sigmoid(x)


class TwoLayerONE(nn.Module):
    def __init__(self, d_embed, length) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(5, d_embed * 2, 7, padding="same")
        # self.bn1 = nn.BatchNorm1d(d_embed * 2)
        # self.conv2 = nn.Conv1d(d_embed * 2, d_embed * 4, 3, padding=1)
        # self.bn2 = nn.BatchNorm1d(d_embed * 4)
        # self.conv3 = nn.Conv1d(d_embed * 4, d_embed, 3, padding=1)
        # self.bn3 = nn.BatchNorm1d(d_embed)
        self.fc0 = nn.Linear(length * d_embed * 2, length * d_embed)
        self.fc1 = nn.Linear(length * d_embed, length * d_embed // 2)
        self.fc2 = nn.Linear(length * d_embed // 2, 1)

    def forward(self, x):
        # x = self.embed(x)
        x = F.one_hot(x, num_classes=5).float()
        x = x.transpose(1, 2)
        # x = torch.relu(self.bn1(self.conv1(x)))
        # x = torch.relu(self.bn2(self.conv2(x)))
        # x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.conv1(x))
        #x = torch.relu(self.conv2(x))
        #x = torch.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x, torch.sigmoid(x)


class Motif(nn.Module):
    def __init__(self, d_embed, length) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(5, d_embed * 2, 7, padding="same")
        # self.bn1 = nn.BatchNorm1d(d_embed * 2)
        # self.conv2 = nn.Conv1d(d_embed * 2, d_embed * 4, 3, padding=1)
        # self.bn2 = nn.BatchNorm1d(d_embed * 4)
        # self.conv3 = nn.Conv1d(d_embed * 4, d_embed, 3, padding=1)
        # self.bn3 = nn.BatchNorm1d(d_embed)
        self.fc0 = nn.Linear(d_embed * 2, 1)
        self.mp = nn.MaxPool1d(length)
        # self.fc1 = nn.Linear(length * d_embed, length * d_embed // 2)
        # self.fc2 = nn.Linear(length * d_embed // 2, 1)

    def forward(self, x):
        # x = self.embed(x)
        x = F.one_hot(x, num_classes=5).float()
        x = x.transpose(1, 2)
        # x = torch.relu(self.bn1(self.conv1(x)))
        # x = torch.relu(self.bn2(self.conv2(x)))
        # x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.conv1(x))
        #x = torch.relu(self.conv2(x))
        #x = torch.relu(self.conv3(x))
        # x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.mp(x).squeeze()
        # print(x.shape)

        # print(x.shape)
        x = self.fc0(x)
        return x, torch.sigmoid(x)