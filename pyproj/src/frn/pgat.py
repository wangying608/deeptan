r"""
This file contains the implementation of the Pooled-GAT.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning as ltn
# from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
# from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from multiprocessing import cpu_count
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from torch.cuda import device_count


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, 20)  # 均值
        self.fc22 = nn.Linear(400, 20)  # 方差

    def forward(self, x):
        x = x.view(-1, input_dim)
        x = torch.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def forward(self, z):
        z = torch.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(z))
        return x_recon


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc5 = nn.Linear(20, 400)
        self.fc6 = nn.Linear(400, input_dim)

    def forward(self, z):
        z = torch.relu(self.fc5(z))
        x_gen = torch.sigmoid(self.fc6(z))
        return x_gen

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc7 = nn.Linear(input_dim, 400)
        self.fc8 = nn.Linear(400, 1)

    def forward(self, x):
        x = x.view(-1, input_dim)
        x = torch.relu(self.fc7(x))
        prob_real_or_fake = torch.sigmoid(self.fc8(x))
        return prob_real_or_fake

# 定义AVAE模型
class AVAE(nn.Module):
    def __init__(self, encoder, decoder, generator, discriminator):
        super(AVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        # 编码
        mu, logvar = self.encoder(x)

        # 从潜在空间采样
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std

        # 解码
        x_recon = self.decoder(z)

        return x_recon, mu, logvar
