"""Preprocessing: imputation, engineered ratio features, geographic cluster similarity, scaling, one-hot."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "median_house_value"
NUMERIC = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
           "population", "households", "median_income"]
CATEGORICAL = ["ocean_proximity"]
SKEWED = ["total_rooms", "total_bedrooms", "population", "households", "median_income"]


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """RBF similarity of each district to k geographic cluster centres (learned with KMeans)."""

    def __init__(self, n_clusters=10, gamma=1.0, random_state=42):
        self.n_clusters, self.gamma, self.random_state = n_clusters, gamma, random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"cluster_{i}_similarity" for i in range(self.n_clusters)]


class Ratio(BaseEstimator, TransformerMixin):
    """First column divided by the second, e.g. bedrooms / rooms."""

    def __init__(self, name="ratio"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, [0]] / X[:, [1]]

    def get_feature_names_out(self, names=None):
        return [self.name]


def ratio_pipeline(name):
    return make_pipeline(SimpleImputer(strategy="median"), Ratio(name), StandardScaler())


def build_preprocessor(engineered=True):
    """engineered=False gives the plain baseline (impute + scale + one-hot)."""
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    if not engineered:
        return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
    log = make_pipeline(SimpleImputer(strategy="median"),
                        FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())
    return ColumnTransformer([
        ("bedrooms_ratio", ratio_pipeline("bedrooms_per_room"), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline("rooms_per_household"), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline("people_per_household"), ["population", "households"]),
        ("log", log, SKEWED),
        ("geo", ClusterSimilarity(n_clusters=10, gamma=1.0), ["latitude", "longitude"]),
        ("cat", cat, CATEGORICAL),
    ], remainder=num)


def make_model(estimator, engineered=True):
    return Pipeline([("prep", build_preprocessor(engineered)), ("model", estimator)])
