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


# Build a Class for GATv2 in PyTorch lightning
