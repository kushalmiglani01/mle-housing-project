"""
This module contains the default arguments and global variabls.
Following arguments and global variables are used

* DOWNLOAD_ROOT
* HOUSING_URL
* PROJECT_ROOT
* HOUSING_PATH
* TRAIN_PATH
* TEST_PATH
* MODEL_PATH

"""

import os

# Global variables related to data loading
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Default Args
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
HOUSING_PATH = os.path.join(PROJECT_ROOT, "datasets", "housing")
TRAIN_PATH = os.path.join(HOUSING_PATH, "train.csv")
TEST_PATH = os.path.join(HOUSING_PATH, "test.csv")
TRAIN_PATH = os.path.join(HOUSING_PATH, "train.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts")
TEST_PATH = os.path.join(HOUSING_PATH, "test.csv")
grid_search_pkl_path = os.path.join(MODEL_PATH, "grid_search_model.pkl")
