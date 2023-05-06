import numpy as np
from housingmodel.custom_logger import *
from sklearn.base import BaseEstimator, TransformerMixin

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
