import os
from argparse import ArgumentParser, Namespace
from train import *

import joblib
import numpy as np
import pandas as pd
from default_args import HOUSING_PATH, MODEL_PATH
from sklearn.metrics import mean_squared_error


def main(housing_path: str = None, model_path: str = None):
    if housing_path is None:
        housing_path = HOUSING_PATH
        print(f"No training_path provided, taking {housing_path}")

    if model_path is None:
        model_path = MODEL_PATH
        print(f"No training_path provided, taking {model_path}")

    test_path = os.path.join(housing_path, "test.csv")

    strat_test_set = pd.read_csv(test_path)

    # laod model
    grid_search_pkl_path = os.path.join(MODEL_PATH, "grid_search_model.pkl")
    grid_search_prep = joblib.load(grid_search_pkl_path)

    # Model prediction
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    final_predictions = grid_search_prep.predict(X_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("final_mse: \t", final_mse)
    print("final_rmse: \t", final_rmse)


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

    args: Namespace = parser.parse_args()

    if args.test_path is None:
        test_path = None
    else:
        test_path = args.test_path

    if args.model_path is None:
        model_path = None
    else:
        model_path = args.model_path

    main(test_path, model_path)
