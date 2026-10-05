import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each category by its relative frequency in the training data.
    Missing values count as their own category; unseen categories map to 0."""
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.maps_ = {c: X[c].fillna("Missing").value_counts(normalize=True).to_dict() for c in self.columns_}
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return np.column_stack([X[c].fillna("Missing").map(self.maps_[c]).fillna(0.0).to_numpy(dtype=float)
                                for c in self.columns_])
    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.columns_])
