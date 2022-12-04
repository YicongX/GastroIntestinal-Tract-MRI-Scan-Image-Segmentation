import argparse
import gc
from uwmgi_preproc import UWMGI
from uwmgi_dataset import UWMGIDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from torch.utils.data import Dataset, DataLoader
from utils import *

from model import *
from schedulers import *
from train import *


parser = argparse.ArgumentParser(description='Load UWMGI.')
parser.add_argument('--model', type=str, default='baseline')
parser.add_argument('--ckpnt', type=str, default=None)
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--fold', type=float, default=5)
parser.add_argument('--fold_selected', type=float, default=1)
parser.add_argument('--img_size', type=int, default=224)
args = parser.parse_args()
data_path = args.data_path
img_size = args.img_size

anno_dir = data_path + 'train.csv'
img_dir = data_path + 'train/'
data = UWMGI(args, anno_dir, img_dir)
df_train = data.df_train
train_ids = data.train_ids
val_ids = data.valid_ids

train_dataset = UWMGIDataset(df_train[df_train.index.isin(train_ids)],img_size)
valid_dataset = UWMGIDataset(df_train[df_train.index.isin(val_ids)],img_size)

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=64,num_workers=4, shuffle=False, pin_memory=True)

# Check mask and input image size
imgs, msks = next(iter(train_loader))
print(imgs.size(), msks.size())

# visulize data
imgs, msks = next(iter(train_loader))
imgs.size(), msks.size()
plot_batch(imgs, msks, size=5)

gc.collect()

model = build_model()
optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR, weight_decay=CONFIG.WEIGHT_DECAY)
scheduler = fetch_scheduler(optimizer)

run = wandb.init(project='uw-maddison-gi-tract', 
                     config={k:v for k, v in dict(vars(CONFIG)).items() if '__' not in k},
                     name=f"model-{CONFIG.MODEL_NAME}",
                     reinit = True
                    )
model, history = run_training(model, optimizer, scheduler,
                                  device=DEVICE,
                                  num_epochs=CONFIG.EPOCHS,
                                  train_loader=train_loader,
                                  valid_loader=valid_loader)
run.finish()
