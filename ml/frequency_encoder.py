import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        self.frequencies_ = {}

        for column in X.columns:
            frequencies = (
                X[column]
                .fillna("__MISSING__")
                .value_counts(normalize=True)
            )

            self.frequencies_[column] = frequencies

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        result = pd.DataFrame(index=X.index)

        for column in X.columns:
            values = (
                X[column]
                .fillna("__MISSING__")
            )

            result[column] = (
                values
                .map(self.frequencies_[column])
                .fillna(0)
            )

        return result.values

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            return self.feature_names_in_

        return np.asarray(
            input_features,
            dtype=object
        )