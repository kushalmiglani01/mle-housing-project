import os
from argparse import ArgumentParser, Namespace
from custom_logger import *
import joblib
import numpy as np
import pandas as pd
from default_args import MODEL_PATH, TRAIN_PATH
from scipy.stats import expon, reciprocal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

### Seting Logger
logger = configure_logger()


# Custom Class to add attributes
class addAttributes(BaseEstimator, TransformerMixin):
    def __init__(
        self, rooms_ix, households_ix, population_ix, bedrooms_ix
    ) -> None:
        super().__init__()
        self.rooms_ix = rooms_ix
        self.households_ix = households_ix
        self.population_ix = population_ix
        self.bedrooms_ix = bedrooms_ix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Adding Atrributes Transformer Started!")
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = (
            X[:, self.population_ix] / X[:, self.households_ix]
        )
        bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
        return np.c_[
            X, rooms_per_household, population_per_household, bedrooms_per_room
        ]


# Custom Transformer for Feature Selection
class featureSelectorRFE(BaseEstimator, TransformerMixin):
    # Initiating the class feature selector with required inputs
    def __init__(
        self,
        support_,
        feature_importances_,
        n_features_,
        k_features_limit=None,
    ) -> None:
        super().__init__()
        self.support_ = support_
        self.feature_importances_ = feature_importances_
        self.n_features_ = n_features_
        self.k_features_limit = k_features_limit

    def fit(self, X, y=None):
        return self

    # Getting the important features and its index using rfecv object
    def transform(self, X, y=None):
        # self.fit(X, y)
        logger.info("Feature Selector Transformer Started!")
        self.features_used_index = [
            i for i, x in enumerate(self.support_) if x
        ]
        arr = self.feature_importances_
        if self.k_features_limit == None:
            k = self.n_features_
        else:
            k = self.k_features_limit
        top_k_indices = np.sort(
            np.argpartition(np.array(arr), -k)[-k:]
        ).tolist()
        indices = [self.features_used_index[i] for i in top_k_indices]
        return X[:, indices]


def main(train_path=None, model_path=None):
    """ """

    if train_path is None:
        train_path = TRAIN_PATH
        logger.info(f"No training_path provided, taking {train_path}")

    if model_path is None:
        model_path = MODEL_PATH
        logger.info(f"No training_path provided, taking {model_path}")

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
    logger.info("Extracting feature using RFE, selecting the best 5 features")
    k_features = 5
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
    grid_search_prep = GridSearchCV(
        single_pipeline,
        param_grid,
        cv=2,
        scoring="neg_mean_squared_error",
        verbose=False,
    )
    logger.info("Finding the best model")
    grid_search_prep.fit(housing, housing_labels)
    grid_search_pkl_path = os.path.join(model_path, "grid_search_model.pkl")
    joblib.dump(grid_search_prep, grid_search_pkl_path)
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
        main(train_path, model_path)
    except Exception as err:
        logger.error(f"Model training failed, Unexpected {err=}, {type(err)=}")
