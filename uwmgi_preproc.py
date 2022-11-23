import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from glob import glob
import os
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold


class UWMGI:
    """UWMGI DATA PREPROCESSING"""

    def __init__(self, args, anno_dir=None, img_dir=None):

        print("Preprocessing Data.")

        df = pd.read_csv(anno_dir)
        # Print original annotation file shape
        print(df.shape)
        # Creating new columns called case, day and slice to store caseid, day9d, slideid
        df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
        df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
        df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

        all_train_images = glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
        x = all_train_images[0].rsplit("/", 4)[0] 

        path_partial_list = []
        for i in range(0, df.shape[0]):
            path_partial_list.append(os.path.join(x,
                                "case"+str(df["case"].values[i]),
                                "case"+str(df["case"].values[i])+"_"+ "day"+str(df["day"].values[i]),
                                "scans",
                                "slice_"+str(df["slice"].values[i])))
        df["path_partial"] = path_partial_list

        path_partial_list = []
        for i in range(0, len(all_train_images)):
            path_partial_list.append(str(all_train_images[i].rsplit("_",4)[0]))
            
        tmp_df = pd.DataFrame()
        tmp_df['path_partial'] = path_partial_list
        tmp_df['path'] = all_train_images

        # Adding the path to images to the dataframe 
        df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])

        # Creating new columns height and width from the path details of images
        df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
        df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))

        # Deleting redundant columns
        del x,path_partial_list,tmp_df
        # Cleaning the dataframe
        df_train = pd.DataFrame({'id':df['id'][::3]})

        # Adding the segementations directly to the class object they belong to
        df_train['large_bowel'] = df['segmentation'][::3].values
        df_train['small_bowel'] = df['segmentation'][1::3].values
        df_train['stomach'] = df['segmentation'][2::3].values

        df_train['path'] = df['path'][::3].values
        df_train['case'] = df['case'][::3].values
        df_train['day'] = df['day'][::3].values
        df_train['slice'] = df['slice'][::3].values
        df_train['width'] = df['width'][::3].values
        df_train['height'] = df['height'][::3].values

        df_train.reset_index(inplace=True,drop=True)
        df_train.fillna('',inplace=True); 
        df_train['count'] = np.sum(df_train.iloc[:,1:4]!='',axis=1).values
        # print cleaned up data shape
        print(df_train.shape)

        # Creating samples, remove sample without segmentation mask
        train_mask = list(df_train[df_train['large_bowel']!=''].index)
        train_mask += list(df_train[df_train['small_bowel']!=''].index)
        train_mask += list(df_train[df_train['stomach']!=''].index)

        df_train=df_train[df_train.index.isin(train_mask)]     
        df_train.reset_index(inplace=True,drop=True)
        # print train data frame shape with no segmentation mask sample removed
        print(df_train.shape)
        self.df_train = df_train
        skf = StratifiedGroupKFold(n_splits=args.fold, shuffle=True, random_state=42)
        for fold, (_, val_idx) in enumerate(skf.split(X=df_train, y=df_train['count'],groups =df_train['case']), 1):
            df_train.loc[val_idx, 'fold'] = fold
            
        df_train['fold'] = df_train['fold'].astype(np.uint8)

        self.train_ids = df_train[df_train["fold"]!=args.fold_selected].index
        self.valid_ids = df_train[df_train["fold"]==args.fold_selected].index

    
    def __len__(self):
        return len(self.df_train)

