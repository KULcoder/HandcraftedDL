"""
Random forest is a nice algorithm as its utilizaed both the complexity of decision tree methods 
and the randomness to balance between bias and variance.

Using sklearn style.
"""

import numpy as np
from joblib import Parallel, delayed # for parallel execution of random forest
from decision_tree import DecisionTree # individual decision tree

class RandomForest:
    def __init__(self, is_classification=True, n_estimators=100, max_depth=2, min_samples_split=1, n_jobs=-1, data_bootstrap_rate=0.5, feature_bootstrap_rate=0.5):
        self.is_classification = is_classification # decide if this is a classification or regression tree
        self.n_estimators = n_estimators # number of trees in the forest
        self.max_depth = max_depth # regularized parameter (max depth of the tree)
        self.min_samples_split = min_samples_split # regularized parameter (min number of samples to split)
        self.n_jobs = n_jobs # number of jobs to run in parallel (decide by the number of cores you have)
        self.data_bootstrap_rate = data_bootstrap_rate # rate of data to bootstrap (with replacement)
        self.feature_bootstrap_rate = feature_bootstrap_rate # rate of features to bootstrap (without replacement)

        self.trees = []
        self.is_trained = False

    def fit(self, X, y):
        if self.is_trained:
            print('The model has been trained.')
            return

        # train the random forest
        self.trees = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_tree)(X, y) for _ in range(self.n_estimators))
        self.is_trained = True

    def _fit_tree(self, X, y):
        tree = DecisionTree(is_classification=self.is_classification, max_depth=self.max_depth, min_samples_split=self.min_samples_split)

        # bootstrap the data
        n_samples = int(X.shape[0] * self.data_bootstrap_rate)
        data_idx = np.random.choice(X.shape[0], n_samples, replace=True)
        X, y = X[data_idx], y[data_idx]

        # bootstrap the features
        n_features = int(X.shape[1] * self.feature_bootstrap_rate)
        feature_index = np.random.choice(X.shape[1], n_features, replace=False)
        X = X[:, feature_index]

        tree.fit(X, y)
        return (tree, feature_index)

    def predict(self, X):
        if not self.is_trained:
            print('The model has not been trained.')
            return
        
        # make prediction
        y_pred = Parallel(n_jobs=self.n_jobs)(delayed(self._predict_tree)(X, tree) for tree in self.trees)
        y_pred = np.array(y_pred).T


        if self.is_classification:
            # aggregate the predictions from all individual trees for each observation
            num_classes = len(np.unique(y_pred))
            def provide_result(row):
                row = np.bincount(row, minlength=num_classes)
                return np.argmax(row)
            return np.apply_along_axis(provide_result, 1, y_pred)
        else:
            return np.mean(y_pred, axis=1)
        
    def _predict_tree(self, X, tree):
        dt = tree[0]
        feature_index = tree[1]
        return dt.predict(X[:, feature_index])