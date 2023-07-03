"""
train.py module trains the model for predicting median house prices. It takes following arguments

usage: train.py [-h] [-t TRAIN_PATH] [-m MODEL_PATH] [-ll LOG_LEVEL] [-lp LOG_PATH] [-cl CONSOLE_LOG]

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN_PATH, --train_path TRAIN_PATH
                        Provide the path for training data
  -m MODEL_PATH, --model_path MODEL_PATH
                        Provide the path for model output
  -ll LOG_LEVEL, --log_level LOG_LEVEL
                        Provide the log level, default is set to debug
  -lp LOG_PATH, --log_path LOG_PATH
                        Provide the log_path if log file is needed, default is set to None
  -cl CONSOLE_LOG, --console_log CONSOLE_LOG
                        select if logging is required in console, default is set to True
"""


import os
import pickle
import sys
from argparse import ArgumentParser, Namespace

import cloudpickle
import joblib
import mlflow
import pandas as pd
from housingmodel.custom_logger import *
from housingmodel.custom_transformers import addAttributes, featureSelectorRFE
from housingmodel.default_args import MODEL_PATH, PROJECT_ROOT, TRAIN_PATH
from scipy.stats import expon, reciprocal
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, PROJECT_ROOT)

### Seting Logger
logger = configure_logger()


def main(train_path=None, model_path=None):
    """Main method for training the model.

    Generate a model pickle file as part of model training
    and save them in the specified output directory

    Parameters
    ----------
    train_path : str
        Full path for train.csv
    model_path : str
        Full path for the pkl file where output will be generated

    Returns
    -------
    None

    Notes
    -----
    This function generates artifacts by applying transformations and training the model to the input data
    and saving the resulting artifacts in the specified output directory.

    Examples
    --------

    train_path = "/datasets/housing/train.csv"
    model_path = "artifacts/grid_search_model.pkl"

    """

    if train_path is None:
        train_path = TRAIN_PATH
        logger.info(f"No training_path provided, taking {train_path}")

    if model_path is None:
        model_path = MODEL_PATH
        logger.info(f"No training_path provided, taking {model_path}")
        grid_search_pkl_path = os.path.join(
            model_path, "grid_search_model.pkl"
        )
    else:
        grid_search_pkl_path = model_path

    # mlflow logging the parameters
    mlflow.log_param("train_path", train_path)
    mlflow.log_param("model_path", model_path)

    # Importing training data
    logger.info("Setting up the training data")
    strat_train_set = pd.read_csv(train_path)

    # Preparing test data for pipeline
    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    # Custom Transformer for adding new attributes
    col_names = "total_rooms", "total_bedrooms", "population", "households"

    rooms_ix, bedrooms_ix, population_ix, households_ix = [
        housing.columns.get_loc(c) for c in col_names
    ]

    # numeric transformation pipeline
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "attribs_adder",
                addAttributes(
                    rooms_ix, households_ix, population_ix, bedrooms_ix
                ),
            ),
            ("std_scaler", StandardScaler()),
        ]
    )

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # data preprocessing pipeline
    logger.info("Setting up the data processing pipeline")
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    housing_prepared = full_pipeline.fit_transform(housing)

    # Feature elimination using RFE
    k_features = 5
    logger.info(
        f"Extracting feature using RFE, selecting the best {k_features} features"
    )
    reg_rf = RandomForestRegressor(random_state=42)
    rfecv = RFECV(
        estimator=reg_rf,
        step=1,
        cv=3,
        n_jobs=-1,
        min_features_to_select=k_features,
        verbose=False,
        importance_getter="feature_importances_",
    )

    rfecv.fit(housing_prepared, housing_labels)

    rfecv_support_ = rfecv.support_
    rfecv_feature_importances_ = rfecv.estimator_.feature_importances_
    rfecv_n_features_ = rfecv.n_features_

    # Model persistence and Model training
    rand_search_path = os.path.join(MODEL_PATH, "rand_search_svm_result.pkl")
    param_distribs = {
        "kernel": ["linear", "rbf"],
        "C": reciprocal(20, 20000),
        "gamma": expon(scale=1.0),
    }

    mlflow.log_param("feature elimnation", "rfe")
    mlflow.log_param("estimator_rfe", "reg_rf")
    mlflow.log_param("min_features_to_select", k_features)
    mlflow.log_param("cv", 3)
    mlflow.log_artifact(rand_search_path)

    logger.info("Tuning the hyperparameters")
    if not os.path.exists(rand_search_path):
        logger.info(
            "Previous model artifact not present, tuning hyperparameters using RandomizedSearchCV"
        )
        svm_regressor = SVR()
        rand_search = RandomizedSearchCV(
            svm_regressor,
            param_distribs,
            n_iter=8,
            cv=2,
            scoring="neg_mean_squared_error",
            verbose=False,
            n_jobs=-1,
            random_state=42,
        )

        rand_search.fit(housing_prepared, housing_labels)

        joblib.dump(
            rand_search, "datasets/artifacts/rand_search_svm_result.pkl"
        )

    rand_search = joblib.load(rand_search_path)

    # Single model pipeline for trained svr hyperparameters
    logger.info(
        "Formulated single pipeline for model using trained hyperparameters"
    )
    single_pipeline = Pipeline(
        [
            ("data_preparation", full_pipeline),
            (
                "feature_selection",
                featureSelectorRFE(
                    rfecv_support_,
                    rfecv_feature_importances_,
                    rfecv_n_features_,
                    k_features,
                ),
            ),
            ("svm_reg", SVR(**rand_search.best_params_)),
        ]
    )

    # Further model exploration and final model selection
    logger.info("Doing further model exploration")
    full_pipeline.named_transformers_["cat"].handle_unknown = "ignore"
    param_grid = [
        {
            "data_preparation__num__imputer__strategy": [
                "mean",
                "median",
                "most_frequent",
            ],
            "feature_selection__k_features_limit": list(
                range(3, rfecv_n_features_ + 1)
            ),
        }
    ]

    mlflow.log_param("parent", "yes")
    try:
        with mlflow.start_run(
            run_name="CHILD_RUN",
            experiment_id=mlflow.get_experiment_by_name(
                exp_name
            ).experiment_id,
            description="child",
            nested=True,
        ) as child_run:
            mlflow.log_param("child", "yes")
            mlflow.sklearn.autolog(log_post_training_metrics=False)
            grid_search_prep = GridSearchCV(
                single_pipeline,
                param_grid,
                cv=2,
                scoring="neg_mean_squared_error",
                verbose=False,
            )
    except:
        with mlflow.start_run(
            run_name="CHILD_RUN",
            description="child",
            nested=True,
        ) as child_run:
            mlflow.log_param("child", "yes")
            mlflow.sklearn.autolog(log_post_training_metrics=False)
            grid_search_prep = GridSearchCV(
                single_pipeline,
                param_grid,
                cv=2,
                scoring="neg_mean_squared_error",
                verbose=False,
            )

    logger.info("Finding the best model")
    grid_search_prep.fit(housing, housing_labels)

    mlflow.log_artifact(grid_search_pkl_path)

    with open(grid_search_pkl_path, "wb") as f:
        pickle.dump([addAttributes, featureSelectorRFE, grid_search_prep], f)
    logger.info(
        f"Model training complete find the pkl at {grid_search_pkl_path}"
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-t",
        "--train_path",
        help="Provide the path for training data",
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
        help="Provide the log_path if log file is needed, default is set to None",
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

    if args.train_path is None:
        train_path = None
    else:
        train_path = args.train_path

    if args.model_path is None:
        model_path = None
    else:
        model_path = args.model_path
    try:
        artifact_path = os.path.join(PROJECT_ROOT, "artifacts")
        print(artifact_path)

        exp_name = f"training_module"
        print(f"file:/{artifact_path}/mlruns")
        mlflow.set_tracking_uri(f"file:{artifact_path}/mlruns")
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(
            experiment_id=mlflow.get_experiment_by_name(exp_name).experiment_id
        ):
            main(train_path, model_path)
    except Exception as err:
        logger.error(f"Model training failed, Unexpected {err=}, {type(err)=}")
        raise
