import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix


class ExNode:
    def __init__(self, size, left=None, right=None):
        self.size = size
        self.left = left
        self.right = right


class InNode:
    def __init__(self, split_att, split_val, left=None, right=None):
        self.split_att = split_att
        self.split_val = split_val
        self.left = left
        self.right = right


class IsolationTree:
    def __init__(self, height_limit, e=0, n_nodes=0, split_att=None, split_val=None):
        self.height_limit = height_limit
        self.root = InNode(split_att, split_val)
        self.e = e
        self.n_nodes = n_nodes
        self.split_att = split_att
        self.split_val = split_val

    def fit(self, X: np.ndarray, improved):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if (self.e >= self.height_limit) or (len(X) <= 1):
                self.root=ExNode(len(X))
                return self.root

        else:
            if not improved:
                att_range = np.arange(X.shape[1])
                self.split_att = random.choice(att_range)
                self.split_val = random.uniform(X[:, self.split_att].min(), X[:, self.split_att].max())

            elif improved:
                att_range = np.arange(X.shape[1])
                att_list = random.sample(list(att_range), 4)

                dist_qp = dict()
                for q in att_list:
                    p = np.random.uniform(X[:, q].min(), X[:, q].max())
                    size = min(len(X[X[:, q] < p]), len(X[X[:, q] > p]))
                    dist_qp[size] = (q, p)
                tuple = dist_qp[min(dist_qp.keys())]
                self.split_att = tuple[0]
                self.split_val = tuple[1]

            X_left = X[X[:, self.split_att] < self.split_val]
            X_right = X[X[:, self.split_att] >= self.split_val]

            left_tree = IsolationTree(self.height_limit, self.e + 1).fit(X_left, improved)
            right_tree = IsolationTree(self.height_limit, self.e + 1).fit(X_right, improved)
            self.n_nodes = self.num_nodes(left_tree) + self.num_nodes(right_tree)
            self.root = InNode(self.split_att, self.split_val,
                               left_tree, right_tree)
            return self.root
    
    def num_nodes(self, tree):
        if tree is None:
            return 0
        num_left = self.num_nodes(tree.left)
        num_right = self.num_nodes(tree.right)
        return 1 + num_left + num_right
        

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        
    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        height_limit = np.ceil(np.log2(self.sample_size))
        for i in range(self.n_trees):
            sub_X = X[np.random.choice(X.shape[0], self.sample_size, replace=False)]
            i_tree = IsolationTree(height_limit)
            i_tree.fit(sub_X, improved)
            self.trees.append(i_tree)
        return self

    def path_length(self, x, tree, e=0) -> np.ndarray:
        """
        Given an instance of observations x_i from X, and an iTree
        compute the path length for the observation x_i the given tree
        """
        while isinstance(tree, InNode):
            if x[tree.split_att] < tree.split_val:
                tree = tree.left
                e += 1
            else:
                tree = tree.right
                e += 1
        path_length_x = e + c_function(tree.size)
        return path_length_x
        
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        c = c_function(self.sample_size)

        path_lengths = []
        for x in X:
            for tree in self.trees:
                path_lengths.append(self.path_length(x, tree.root))

        path_matrix = np.asarray(path_lengths).reshape(len(X), -1)
        Eh = path_matrix.mean(axis=1)

        s = 2 ** (-1.0 * Eh / c)
        return s

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores >= threshold).astype(int)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        A shorthand for calling anomaly_score() and predict_from_anomaly_scores().
        """
        scores = self.anomaly_score(X)
        prediction = self.predict_from_anomaly_scores(scores, threshold)
        return prediction


def c_function(n):
    if n > 2:
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    elif n == 2:
        return 1
    else:
        return 0


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1
    step = 0.01
    while threshold > 0:
        y_hat = (scores >= threshold).astype(int)
        TN, FP, FN, TP = confusion_matrix(y, y_hat).flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR < desired_TPR:
            threshold -= step
        else:
            break
    return threshold, FPR

