# Data modules
from loguru import logger
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from utils import FeatUtils

class HarDataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir_path: str, num_workers: int=8, feat_shape=(128, 9), batch_size: int=64,
                 train_val_ratio=0.8, simple_loader: bool = False, normalize="minmax", scaler=None):
        # For LstmAutoEncoder, feat_shape = (383, 1)
        # For ConvAutoEncoder, feat_shape = (1, 383)
        # n_train_sampple = 0 -> use all data
        super().__init__()
        
        self.data_dir_path = data_dir_path
        self.train_val_ratio = train_val_ratio
        self.num_workers = num_workers # For DataLoader parameter
        self.feat_shape = feat_shape
        self.batch_size = batch_size
        self.simple_loader = simple_loader

        # Load train features
        X_train, y_train = FeatUtils.load_dataset_group("train", data_dir_path)
        # Decrease label's value by one to match the index of prediction outputs
        y_train["label"] = y_train["label"] - 1
        
        # Show class stat
        n_row = len(y_train)
        for i in np.unique(y_train):
            n_label = len(y_train.loc[y_train["label"] == i])
            print(f"(Train) Class {i}: {n_label} rows {(n_label / n_row) * 100}%")

        X_train, X_valid, y_train, y_valid = FeatUtils.make_split_feature(
            X_train, y_train, prep_func=None, split_frac=train_val_ratio)
        
        # Load test features
        X_test, y_test = FeatUtils.load_dataset_group("test", data_dir_path)
        # Decrease label's value by one to match the index of prediction outputs
        y_test["label"] = y_test["label"] - 1
        X_test = np.array(X_test.astype("float32")).reshape(-1, 128, 9, order="F")

        # Show class stat
        n_row = len(y_test)
        for i in np.unique(y_test):
            n_label = len(y_test.loc[y_test["label"] == i])
            print(f"(Test) Class {i}: {n_label} rows {(n_label / n_row) * 100}%")

        y_test = np.array(y_test).squeeze()
            
        if normalize == "std":
            logger.debug("Normalization method: StandardScaler")
            self.scaler = StandardScaler()
        elif normalize == "minmax":
            logger.debug("Normalization method: MinMaxScaler")
            self.scaler = MinMaxScaler()
        elif normalize is None:
            logger.debug("Normalization method is not set.")
            self.scaler = None
        else:
            logger.warning(f"Unsupported normalization method: {normalize}, fallback to no normalization")
            self.scaler = None
            
            
        if scaler is not None:
            # Override by pre-loaded scaler
            logger.debug(f"Use pre-loaded scaler: {scaler}")
            self.scaler = scaler
        if self.scaler is not None:
            # Normalization is selected. Fit the scaler with training data
            X_train = np.array(X_train).reshape(-1, 1)
            self.scaler.fit(X_train)
        
        if self.scaler is not None:
            # Normalization
            logger.debug("Normalize features.")
            X_train = self.scaler.transform(np.array(X_train).reshape(-1, 1)).reshape(-1, feat_shape[0], feat_shape[1])
            X_valid = self.scaler.transform(np.array(X_valid).reshape(-1, 1)).reshape(-1, feat_shape[0], feat_shape[1])
            X_test = self.scaler.transform(np.array(X_test).reshape(-1, 1)).reshape(-1, feat_shape[0], feat_shape[1])
        else:
            # No normalization
            X_train = np.array(X_train).reshape(-1, feat_shape[0], feat_shape[1])
            X_valid = np.array(X_valid).reshape(-1, feat_shape[0], feat_shape[1])
            X_test = np.array(X_test).reshape(-1, feat_shape[0], feat_shape[1])
            

        self.train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        self.val_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
        self.test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        
        del X_train, y_train, X_valid, y_valid, X_test, y_test
    
    def get_scaler(self):
        return self.scaler
    
    def train_dataloader(self):
        # Only train_loader's shuffle should be turned on.
        if self.simple_loader:
            return DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size, drop_last=True)
        else:
            return DataLoader(self.train_data, shuffle=True, num_workers=self.num_workers, 
                              persistent_workers=True, pin_memory=True, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self):
        if self.simple_loader:
            return DataLoader(self.val_data, shuffle=False, batch_size=self.batch_size, drop_last=True)
        else:
            return DataLoader(self.val_data, shuffle=False, num_workers=self.num_workers, 
                              persistent_workers=True, pin_memory=True, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        if self.simple_loader:
            return DataLoader(self.test_data, shuffle=False, batch_size=self.batch_size, drop_last=True)
        else:
            return DataLoader(self.test_data, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True, batch_size=self.batch_size, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True, batch_size=self.batch_size, drop_last=True)