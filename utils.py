import os
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
    def make_train_valid_test_feature(cls, X, y, prep_func=None, norm=False, split_frac=0.8):
        """Make features from loaded dataframes for train/validate/test
        Data are divided to a train set by a ratio of split_frac.
        Then, the remainings are divided to a validation set and a test set equally (half by half).
        """
        # Pre-process features here...
        if prep_func is not None:
            X = prep_func(X)
        
        # Reshape to fit with 128 steps and 9 features
        X = np.array(X.astype("float32")).reshape(-1, 128, 9, order="F")
        y = np.array(y).squeeze()

        # Use `stratify` to preserve class distribution
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_frac, random_state=88, stratify=y)

        if norm:
            mean, std = cls.compute_mean_std(X_train)

            X_train = cls.normalize_feature(X_train, mean, std)
            X_test = cls.normalize_feature(X_test, mean, std)
        
        # Make validation set and test 50:50
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=88, stratify=y_test)

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    @classmethod
    def make_dataloaders(cls, X_train, X_valid, X_test, y_train, y_valid, y_test, batch_size=128):
        # Create Tensor datasets
        train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        valid_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
        test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        # Make sure to SHUFFLE your data
        # NOTE: Drop last to prevent a mismatched size of hidden state
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

        return train_loader, valid_loader, test_loader