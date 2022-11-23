import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *
from uwmgi_preproc import UWMGI

class UWMGIDataset(Dataset):
    def __init__(self, df,img_size, subset="train"):
        
        self.df = df
        self.subset = subset
        self.img_size =img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        masks = np.zeros((self.img_size,self.img_size, 3), dtype=np.float32)
        img_path=self.df['path'].iloc[index]
        w=self.df['width'].iloc[index]
        h=self.df['height'].iloc[index]
        img = load_img(self.img_size, img_path)
        if self.subset == 'train':
            for k,j in zip([0,1,2],["large_bowel","small_bowel","stomach"]):
                rles=self.df[j].iloc[index]
                mask = rle_decode(rles, shape=(h, w, 1))
                mask = cv2.resize(mask, (self.img_size,self.img_size))
                masks[:,:,k] = mask
        
        masks = masks.transpose(2, 0, 1)
        img = img.transpose(2, 0, 1)

        if self.subset == 'train': 
            return torch.tensor(img), torch.tensor(masks)
        else: 
            return torch.tensor(img)