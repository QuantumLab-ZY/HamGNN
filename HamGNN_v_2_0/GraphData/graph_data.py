"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-07 20:44:01 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 16:24:33
 */
"""
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Union, Callable
import numpy as np
from torch.utils.data import random_split, Subset
import os

"""
graph_data_module inherits pl.lightningDatamodule to implement the dataset class,
which divides the dataset and builds the dataset loader.  
"""


class graph_data_module(pl.LightningDataModule):
    def __init__(self, dataset: Union[list, tuple, np.array] = None,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.2,
                 batch_size: int = 300,
                 val_batch_size: int = None,
                 test_batch_size: int = None,
                 split_file : str = None):
        super(graph_data_module, self).__init__()
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.split_file = split_file
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or self.val_batch_size

    def setup(self, stage=None):
        """
        Split dataset into training, validation and test sets.
        """
        if self.split_file is not None and os.path.exists(self.split_file):
            print(f"Split dataset by {self.split_file}.")
            S = np.load(self.split_file )
            train_idx = S["train_idx"].tolist()
            val_idx = S["val_idx"].tolist()
            test_idx = S["test_idx"].tolist()
            self.train_data = Subset(self.dataset, indices=train_idx)
            self.val_data = Subset(self.dataset, indices=val_idx)
            self.test_data = Subset(self.dataset, indices=test_idx)
        else:
            if stage == 'fit' or stage is None:
                random_state = np.random.RandomState(seed=42)
                length = len(self.dataset)
                num_train = round(self.train_ratio * length)
                num_val = round(self.val_ratio * length)
                num_test = round(self.test_ratio * length)
                perm = list(random_state.permutation(np.arange(length)))
                train_idx = perm[:num_train]
                val_idx = perm[num_train:num_train+num_val]
                test_idx = perm[-num_test:]
                self.train_data = [self.dataset[i] for i in train_idx]
                self.val_data = [self.dataset[i] for i in val_idx]
                self.test_data = [self.dataset[i] for i in test_idx]
            if stage == 'test':
                self.test_data = self.dataset

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.test_batch_size, pin_memory=True)
