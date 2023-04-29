import os
import tarfile
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from default_args import HOUSING_PATH, HOUSING_URL
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit


# Data loading
def fetch_housing_data(housing_path, housing_url=HOUSING_URL):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(path):
    csv_path = os.path.join(path, "housing.csv")
    return pd.read_csv(csv_path)


# Main function
def main(path: str = None):
    if path is None:
        path = HOUSING_PATH
        print(f"No path provided, taking {path}")

    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")

    # downloading the data
    fetch_housing_data(path)
    housing = load_housing_data(path)

    # Creating income category for stratified shuffle sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Dropping the income category column
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Save the data to train.csv and test.csv
    strat_train_set.to_csv(train_path, index=False, sep=",")
    strat_test_set.to_csv(test_path, index=False, sep=",")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        help="Provide the path for data download",
        default=None,
    )
    args: Namespace = parser.parse_args()
    if args.path is None:
        path = None
    else:
        path = args.path
    main(path)
