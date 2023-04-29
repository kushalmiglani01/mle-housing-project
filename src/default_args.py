import os

# Global variables related to data loading
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Default Args
HOUSING_PATH = os.path.join("../datasets", "housing")
TRAIN_PATH = os.path.join(HOUSING_PATH, "train.csv")
TEST_PATH = os.path.join(HOUSING_PATH, "test.csv")
TRAIN_PATH = os.path.join(HOUSING_PATH, "train.csv")
MODEL_PATH = os.path.join("..", "artifacts")
TEST_PATH = os.path.join(HOUSING_PATH, "test.csv")
grid_search_pkl_path = os.path.join(MODEL_PATH, "grid_search_model.pkl")
