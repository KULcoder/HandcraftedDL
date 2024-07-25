import numpy as np

"""
This implementation doesn't take care of discrete or categorical features:
It is considering every feature to be continuous.

Pure python implementation by hand, not efficient.

How to use:
Create a decision tree object, fit and predict on data.

"""

class node:
    # the node class to represent decisions and leaf nodes
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf

class criterion:
    # the criterion class to decide best split
    def __init__(self, type):
        self.type = type

    def __call__(self, y):
        if self.type == 'gini':
            return self.gini(y)
        elif self.type == 'variance':
            return self.variance(y)
        
    def gini(self, y):
        # calculate gini impurity for multi-class classification
        # y: numpy array of shape (n_samples, )
        # return: gini impurity

        # this might be a slow implementation, use numpy all the way might be faster

        n = len(y)
        if n == 0:
            return 0
        classes = np.unique(y)
        gini = 0
        for c in classes:
            gini += (np.sum(y == c) / n) ** 2
        return 1 - gini
    
    def variance(self, y):
        # calculate variance for regression
        # y: numpy array of shape (n_samples, )
        # return: variance
        return np.var(y)
    
class DecisionTree:
    def __init__(self, is_classification=True, max_depth=10, min_samples_split=2):
        # parameters
        self.is_classification = is_classification # decide if this is a classification or regression tree
        self.max_depth = max_depth # regularized parameter (max depth of the tree)
        self.min_samples_split = min_samples_split # regularized parameter (min number of samples to split)
        self.criterion = criterion('gini') if self.is_classification else criterion('variance')

        self.root = None
        self.is_trained = False

    def fit(self, X, y):
        if self.is_trained:
            print('This model is already trained')
            return
        
        self.root = self._grow_tree(X, y)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            print('This model is not trained yet')
            return
        
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _grow_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(y) < self.min_samples_split:
            # is reach max depth or number of samples is less than min_samples_split
            return node(value=self._leaf_value(y), is_leaf=True)
        
        # find the best split
        feature, threshold = self._find_best_split(X, y)
        if feature is None:
            # cannot find a split
            return node(value=self._leaf_value(y), is_leaf=True)
        
        # split the data
        left_idx, right_idx = self._split(X[:, feature], threshold)
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return node(feature=feature, threshold=threshold, left=left, right=right)
    
    def _find_best_split(self, X, y):
        best_criterion = np.inf
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            # consider every feature
            for threshold in X[:, feature]:
                # consider every point as a threshold
                left_idx, right_idx = self._split(X[:, feature], threshold)
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                criterion = self._criterion(y[left_idx], y[right_idx])
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def _criterion(self, y_left, y_right):
        # calculate the weighted criterion
        n = len(y_left) + len(y_right)
        return (len(y_left) / n) * self.criterion(y_left) + (len(y_right) / n) * self.criterion(y_right)
    
    def _split(self, feature, threshold):
        # split the data into left and right
        left_idx = np.argwhere(feature <= threshold).flatten()
        right_idx = np.argwhere(feature > threshold).flatten()
        return left_idx, right_idx
    
    def _leaf_value(self, y):
        # calculate the leaf value
        if self.is_classification:
            return np.argmax(np.bincount(y)) # for classification, return the most common class
        else:
            return np.mean(y) # for regression, return the mean
        
    def _traverse_tree(self, x, node):
        # traverse the tree to make prediction
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)