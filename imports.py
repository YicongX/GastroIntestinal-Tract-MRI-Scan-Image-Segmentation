# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummaryX import summary
from torch.cuda import amp

import copy
from collections import defaultdict

import numpy as np

# supporting
import gc
import time
import wandb
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import warnings
from IPython import display as ipd
warnings.filterwarnings('ignore')

# external libraries
import segmentation_models_pytorch as smp

# configuration
from configuration import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)

if CONFIG.USE_WANDB:
    wandb.login(key="") # TODO: Add your wandb key


