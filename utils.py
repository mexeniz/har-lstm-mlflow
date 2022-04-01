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
    def save_feat_scaler(scaler, scaler_path):
        logger.info(f"Saving a scaler: scaler={scaler} path={scaler_path}")
        pickle.dump(scaler, open(scaler_path, "wb"))
        
    @staticmethod
    def load_feat_scaler(scaler_path):
        logger.info(f"Loading a scaler: path={scaler_path}")
        scaler = pickle.load(open(scaler_path, "rb"))
        return scaler