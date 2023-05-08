import numpy as np
from housingmodel.custom_logger import *
from sklearn.base import BaseEstimator, TransformerMixin

logger = configure_logger()


# Custom Class to add attributes
class addAttributes(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add new attributes to a feature matrix.

    Parameters
    ----------
    rooms_ix : int
        Index of the column containing the number of rooms in each district.
    households_ix : int
        Index of the column containing the number of households in each district.
    population_ix : int
        Index of the column containing the number of inhabitants in each district.
    bedrooms_ix : int
        Index of the column containing the number of bedrooms in each district.

    Methods
    -------
    fit(X[, y]) -> self
        Fit transformer to the data.

    transform(X) -> ndarray
        Transform the data by adding new attributes.

    Notes
    -----
    This transformer assumes that the input data is a numpy ndarray or a pandas DataFrame.

    """

    def __init__(
        self, rooms_ix, households_ix, population_ix, bedrooms_ix
    ) -> None:
        super().__init__()
        self.rooms_ix = rooms_ix
        self.households_ix = households_ix
        self.population_ix = population_ix
        self.bedrooms_ix = bedrooms_ix

    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,), default=None
            The target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Transform the input data by adding new attributes based on the specified indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        np.ndarray
            Numpy array containing the original input data with additional
            attributes added based on the specified indices.
        """
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
    """
    Custom Transformer for Feature Selection using Recursive Feature Elimination with Cross-Validation

    Parameters:
    -----------
    support_: array-like of shape (n_features,)
        The mask of selected features.
    feature_importances_: array-like of shape (n_features,)
        The feature importances.
    n_features_: int
        The total number of features in the input data.
    k_features_limit: int, optional
        The limit on the maximum number of features to select. If None, it selects all features with non-zero importance.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer on the input data.

    transform(X, y=None)
        Select the most important features based on their importances.

    Notes
    -----
    This transformer assumes that the input data is a numpy ndarray or a pandas DataFrame.

    """

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
        """
        Fit the transformer on the input data.
        Parameters:
        -----------
        X: array-like of shape (n_samples, n_features)
            The input data to be transformed.
        y: array-like of shape (n_samples,), optional
            The target values (class labels in classification, real numbers in regression). This parameter is not used.

        Returns:
        --------
        self: object
        """
        return self

    # Getting the important features and its index using rfecv object
    def transform(self, X, y=None):
        """
        Fit the transformer on the input data.

        transform(X, y=None)
            Select the most important features based on their importances.
            Parameters:
            -----------
            X: array-like of shape (n_samples, n_features)
                The input data to be transformed.
            y: array-like of shape (n_samples,), optional
                The target values (class labels in classification, real numbers in regression). This parameter is not used.

        Returns:
        --------
        X_transformed: array-like of shape (n_samples, k_features)
            The transformed input data with the k most important features.
        """
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
