import os
import pickle
import re

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

class FeatUtils():
    # These load functions are modified based on https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/.
    @staticmethod
    def load_file(filepath):
        "load a single file as a dataframe"
        df = pd.read_csv(filepath, header=None, delim_whitespace=True)
        return df

    @classmethod
    def load_group(cls, filenames, prefix=""):
        """load a list of files into a 3D array of [samples, timesteps, features]"""
        data_dict = {}
        for filename in filenames:
            col = re.sub(r"(_train.txt)|(_test.txt)", "", filename)
            data_dict[col] = cls.load_file(os.path.join(prefix, filename))
        df = pd.concat(data_dict, axis=1)
        return df


    @classmethod
    def load_dataset_group(cls, group, prefix=""):
        """load a dataset group, such as train or test"""
        dirpath = os.path.join(prefix, group, "Inertial Signals")
        # load all 9 files as a single array
        filenames = list()
        for feat_type in ["total_acc", "body_acc", "body_gyro"]:
            for axis in ["x", "y", "z"]:
                filenames.append(f"{feat_type}_{axis}_{group}.txt")
        # load input data
        X = cls.load_group(filenames, prefix=dirpath)
        # load class output
        y = cls.load_file(os.path.join(prefix, group, f"y_{group}.txt"))
        y.rename({0:"label"}, axis=1, inplace=True)
        return X, y
    
    @classmethod
    def make_split_feature(cls, X, y, prep_func=None, norm=False, split_frac=0.8):
        """Make features from loaded dataframes for train/validate
        Data are divided to a train/validate set by a ratio of split_frac.
        """
        # Pre-process features here...
        if prep_func is not None:
            X = prep_func(X)
        
        # Reshape to fit with 128 steps and 9 features
        X = np.array(X.astype("float32")).reshape(-1, 128, 9, order="F")
        y = np.array(y).squeeze()

        # Use `stratify` to preserve class distribution
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=split_frac, random_state=88, stratify=y)

        return X_train, X_valid, y_train, y_valid
    
    @staticmethod
    def standard_normalize(X, means, stds):
        """Feature-wise"""
        X = X.reshape((-1, 9))
        return ((X - means) / stds).reshape((-1, 128, 9))
        
    @staticmethod
    def min_max_normalize(X, max_vals, min_vals):
        """Feature-wise"""
        X = X.reshape((-1, 9))
        return ((X - min_vals) / (max_vals - min_vals)).reshape((-1, 128, 9))
    
    @classmethod
    def normalize_feature(cls, X_train, X_valid, X_test, method="minmax"):
        base_data = np.concatenate((X_train, X_valid), axis=0)
        base_data = base_data.reshape((-1, 9))
        if method == "minmax":
            max_vals = base_data.max(axis=0)
            min_vals = base_data.min(axis=0)
            
            norm_X_train = cls.min_max_normalize(X_train, max_vals, min_vals)
            norm_X_valid = cls.min_max_normalize(X_valid, max_vals, min_vals)
            norm_X_test = cls.min_max_normalize(X_test, max_vals, min_vals)
        elif method == "standard":
            means = base_data.mean(axis=0)
            stds = base_data.std(axis=0)
            
            norm_X_train = cls.standard_normalize(X_train, means, stds)
            norm_X_valid = cls.standard_normalize(X_valid, means, stds)
            norm_X_test = cls.standard_normalize(X_test, means, stds)
        else:
            raise ValueError(f"parameter value is invalid: method={method} expected=['minmax','mean']")
        
        return norm_X_train, norm_X_valid, norm_X_test
    
    @classmethod
    def make_dataloader(cls, X, y, batch_size=128):
        # Create Tensor datasets
        data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

        # Make sure to SHUFFLE your data
        # NOTE: Drop last to prevent a mismatched size of hidden state
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, drop_last=True)

        return data_loader

    @staticmethod
    def save_feat_scaler(scaler, scaler_path):
        logger.info(f"Saving a scaler: scaler={scaler} path={scaler_path}")
        pickle.dump(scaler, open(scaler_path, "wb"))
        
    @staticmethod
    def load_feat_scaler(scaler_path):
        logger.info(f"Loading a scaler: path={scaler_path}")
        scaler = pickle.load(open(scaler_path, "rb"))
        return scaler