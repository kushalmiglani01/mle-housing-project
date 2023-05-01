import os
import tarfile
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from custom_logger import *
from default_args import HOUSING_PATH, HOUSING_URL
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

### Seting Logger
logger = configure_logger()


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
        logger.info(f"No path provided, taking {path}")

    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")

    # downloading the data
    fetch_housing_data(path)
    housing = load_housing_data(path)

    logger.info(
        "Data Downloaded, creating income category for stratified split"
    )

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

    logger.info("Completed the split")
    # Dropping the income category column
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    logger.info("Saving the data")

    # Save the data to train.csv and test.csv
    strat_train_set.to_csv(train_path, index=False, sep=",")
    strat_test_set.to_csv(test_path, index=False, sep=",")

    logger.info(f"Training data saved in {train_path}")
    logger.info(f"Training data saved in {test_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        help="Provide the path for data download",
        default=None,
    )

    parser.add_argument(
        "-ll",
        "--log_level",
        help="Provide the log level, default is set to debug",
        default="DEBUG",
        type=str,
    )

    parser.add_argument(
        "-lp",
        "--log_path",
        help="Provide the full absolute log_path if log file is needed, default is set to None",
        default=None,
        type=str,
    )

    parser.add_argument(
        "-cl",
        "--console_log",
        help="select if logging is required in console, default is set to True",
        default=True,
        type=bool,
    )

    args: Namespace = parser.parse_args()

    log_level = args.log_level
    log_path = args.log_path
    console_log = args.console_log

    if not log_path is None:
        base_name = os.path.basename(__file__).split(".")[0]
        log_path = os.path.join(os.path.abspath(log_path), f"{base_name}.log")

    # Overriding default logger config
    logger = configure_logger(
        log_file=log_path, console=console_log, log_level=log_level
    )

    logger.info(f"log_path: {log_path}")

    if args.path is None:
        path = None
    else:
        path = args.path
    try:
        main(path)
    except Exception as err:
        logger.error(f"Model training failed, Unexpected {err=}, {type(err)=}")
