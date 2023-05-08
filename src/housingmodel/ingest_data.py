"""
ingest_data.py downloads the housing.tgz which contatins the datasets and perform a 
stratified split to generate test.csv and train.csv

usage: ingest_data.py [-h] [-p PATH] [-ll LOG_LEVEL] [-lp LOG_PATH] [-cl CONSOLE_LOG]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Provide the path for data download
  -ll LOG_LEVEL, --log_level LOG_LEVEL
                        Provide the log level, default is set to debug
  -lp LOG_PATH, --log_path LOG_PATH
                        Provide the full absolute log_path if log file is needed, default is set to None
  -cl CONSOLE_LOG, --console_log CONSOLE_LOG
                        select if logging is required in console, default is set to True

"""

import os
import tarfile
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from housingmodel.custom_logger import *
from housingmodel.default_args import HOUSING_PATH, HOUSING_URL
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

### Seting Logger
logger = configure_logger()


# Data loading
def fetch_housing_data(housing_path, housing_url=HOUSING_URL):
    """
    Download the housing dataset from the given URL and save it to the specified directory.

    Parameters
    ----------
    housing_path : str
        The directory path where the housing dataset should be downloaded.

    housing_url : str
        The URL from which the housing dataset should be downloaded.

    Returns
    -------
    None

    Notes
    -----
    This function will create the specified directory if it does not already exist.

    Examples
    --------
    >>> housing_path = 'datasets/housing'
    >>> housing_url = 'https://example.com/housing.tgz'
    >>> fetch_housing_data(housing_path, housing_url)
    """

    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(path):
    """
    Read a CSV file from the given path and return it as a pandas dataframe.

    Parameters
    ----------
    path : str
        The path to the CSV file.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe containing the data from the CSV file.
    """
    csv_path = os.path.join(path, "housing.csv")
    return pd.read_csv(csv_path)


# Main function
def main(path: str = None):
    """
    Downloads the housing.tgz data and creates a train test split.

    Parameters
    ----------
    path : str
        Path to directory where the data will be downloaded.

    Returns
    -------
    None

    Notes
    -----
    This function downloads the housing.tgz file from the internet and extracts the
    housing.csv file. It then reads in the data and creates a train-test split using a
    80-20 split ratio. The split data is then saved as train.csv and test.csv files in
    the same directory as the original data.

    """

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
        logger.error(f"data gathering failed, Unexpected {err=}, {type(err)=}")
