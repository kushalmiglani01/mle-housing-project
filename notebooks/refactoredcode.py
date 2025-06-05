import os
import tarfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import expon, reciprocal
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

# Global variables related to data loading
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# Data loading
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()


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


# Calculating and comparing sampling error
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()
compare_props["Rand. %error"] = (
    100 * compare_props["Random"] / compare_props["Overall"] - 100
)
compare_props["Strat. %error"] = (
    100 * compare_props["Stratified"] / compare_props["Overall"] - 100
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Basic EDA
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

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


class addAttributes(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
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


# numeric transformation pipeline
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", addAttributes()),
        ("std_scaler", StandardScaler()),
    ]
)


housing_num = housing.drop("ocean_proximity", axis=1)


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# data preprocessing pipeline
full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ]
)


housing_prepared = full_pipeline.fit_transform(housing)

# Feature elimination using RFE
k_features = 5
reg_rf = RandomForestRegressor(random_state=42)
rfecv = RFECV(
    estimator=reg_rf,
    step=1,
    cv=3,
    n_jobs=-1,
    min_features_to_select=k_features,
    verbose=2,
    importance_getter="feature_importances_",
)

rfecv.fit(housing_prepared, housing_labels)

rfecv_support_ = rfecv.support_
rfecv_feature_importances_ = rfecv.estimator_.feature_importances_
rfecv_n_features_ = rfecv.n_features_

# Data preprocessing and feature selection pipeline
preparation_and_feature_selection_pipeline = Pipeline(
    [
        ("preparation", full_pipeline),
        (
            "feature_selection",
            featureSelectorRFE(
                rfecv_support_,
                rfecv_feature_importances_,
                rfecv_n_features_,
                k_features,
            ),
        ),
    ]
)

# Model persistence and Model training
rand_search_path = "datasets/artifacts/rand_search_svm_result.pkl"
param_distribs = {
    "kernel": ["linear", "rbf"],
    "C": reciprocal(20, 20000),
    "gamma": expon(scale=1.0),
}


if not os.path.exists(rand_search_path):
    svm_regressor = SVR()
    rand_search = RandomizedSearchCV(
        svm_regressor,
        param_distribs,
        n_iter=8,
        cv=2,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )

    rand_search.fit(housing_prepared, housing_labels)

    joblib.dump(rand_search, "datasets/artifacts/rand_search_svm_result.pkl")

rand_search = joblib.load(rand_search_path)

# Single model pipeline for trained svr hyperparameters
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
    verbose=2,
)
grid_search_prep.fit(housing, housing_labels)


# Model prediction
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = grid_search_prep.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("final_mse: \t", final_mse)
print("final_rmse: \t", final_rmse)
