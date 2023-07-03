"""
score.py module evaluates the performance of the ML model on a test dataset.

usage: score.py [-h] [-t TEST_PATH] [-m MODEL_PATH] [-ll LOG_LEVEL] [-lp LOG_PATH] [-cl CONSOLE_LOG]

optional arguments:
  -h, --help            show this help message and exit
  -t TEST_PATH, --test_path TEST_PATH
                        Provide the path for testing data
  -m MODEL_PATH, --model_path MODEL_PATH
                        Provide the path for model output
  -ll LOG_LEVEL, --log_level LOG_LEVEL
                        Provide the log level, default is set to debug
  -lp LOG_PATH, --log_path LOG_PATH
                        Provide the full absolute log_path if log file is needed, default is set to None
  -cl CONSOLE_LOG, --console_log CONSOLE_LOG
                        select if logging is required in console, default is set to True
"""


import os
import sys
from argparse import ArgumentParser, Namespace

import mlflow
import numpy as np
import pandas as pd
from housingmodel.custom_logger import *

# from housingmodel.custom_transformers import addAttributes, featureSelectorRFE
from housingmodel.default_args import HOUSING_PATH, MODEL_PATH, PROJECT_ROOT
from sklearn.metrics import mean_squared_error

sys.path.insert(0, PROJECT_ROOT)

### Seting Logger
logger = configure_logger()


def main(test_path: str = None, model_path: str = None):
    """Main method for scoring the model.

    Generate model score as part of the testing of the model

    Parameters
    ----------
    test_path : str
        Full path for test.csv
    model_path : str
        Full path for the pkl file where output will be generated

    Returns
    -------
    Tuple[float, float]
        A tuple containing the root mean squared error (RMSE) and the mean squared error (MSE)
        between the model predictions and the ground truth values on the test dataset.

    Notes
    -----
    Evaluate the performance of a machine learning model on a test dataset.

    Examples
    --------

    train_path = "/datasets/housing/test.csv"
    model_path = "artifacts/grid_search_model.pkl"

    """

    if test_path is None:
        housing_path = HOUSING_PATH
        test_path = os.path.join(housing_path, "test.csv")
        logger.info(f"No test data path provided, taking {test_path}")
    else:
        test_path = test_path

    if model_path is None:
        model_path = MODEL_PATH
        grid_search_prep_path = os.path.join(
            model_path, "grid_search_model.pkl"
        )
        logger.info(f"No model path provided, taking {model_path}")
    else:
        grid_search_prep_path = model_path

    strat_test_set = pd.read_csv(test_path)

    print(model_path)
    # laod model
    import pickle

    print(grid_search_prep_path)
    with open(grid_search_prep_path, "rb") as f:
        addAttributes, featureSelectorRFE, grid_search_prep = pickle.load(f)
    # grid_search_prep = joblib.load(grid_search_pkl_path)

    # Model prediction
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    final_predictions = grid_search_prep.predict(X_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    logger.info(f"final_mse:    {final_mse}")
    logger.info(
        f"final_rmse:   {final_rmse}",
    )
    # mlflow logging the parameters
    mlflow.log_param("test_path", test_path)
    mlflow.log_param("model_path", model_path)
    # mlflow logging the metrics
    mlflow.log_metric("final_rmse", final_rmse)
    mlflow.log_metric("final_mse", final_mse)

    return final_mse, final_rmse


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--test_path",
        help="Provide the path for testing data",
        default=None,
    )

    parser.add_argument(
        "-m",
        "--model_path",
        help="Provide the path for model output",
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

    if args.test_path is None:
        test_path = None
    else:
        test_path = args.test_path

    if args.model_path is None:
        model_path = None
    else:
        model_path = args.model_path

    try:
        from housingmodel.train import addAttributes

        artifact_path = os.path.join(PROJECT_ROOT, "artifacts")
        print(artifact_path)

        exp_name = f"evaluation_module"
        print(f"file:/{artifact_path}/mlruns")
        mlflow.set_tracking_uri(f"file:{artifact_path}/mlruns")
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(
            experiment_id=mlflow.get_experiment_by_name(exp_name).experiment_id
        ):
            main(test_path, model_path)
    except Exception as err:
        logger.error(f"Model scoring failed, Unexpected {err=}, {type(err)=}")
        raise
