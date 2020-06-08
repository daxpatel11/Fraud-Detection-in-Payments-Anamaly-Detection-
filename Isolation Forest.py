import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix

class IsolationTreeEnsemble:
    """
    create an ensemble of IsolationTree objects and store them in a list: self.trees.
    """
    def __init__(self, sample_size, n_trees=10):
        """
        input:
            sample_size : Number of instances in one tree
            n_trees : Number of trees
        """
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = np.log2(sample_size)
        self.trees = []

    def fit(self, X:np.ndarray):
        """
        input:
            X : dataset
            n_trees : Number of trees  
        """
        if isinstance(X, pd.DataFrame):
            '''
            Convert DataFrames to ndarray objects.
            '''
            X = X.values
            len_x = len(X)
            col_x = X.shape[1]
            self.trees = []
            
        for i in range(self.n_trees):
            sample_idx = random.sample(list(range(len_x)), self.sample_size)
            temp_tree = IsolationTree(self.height_limit, 0).fit(X[sample_idx, :])
            self.trees.append(temp_tree)

        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        input:
            X : dataset
        Compute the path length for x_i using every tree in self.trees 
        then compute the average for each x_i.
        """
        pl_vector = []
        if isinstance(X, pd.DataFrame):
            X = X.values

        for x in (X):
            pl = np.array([path_length_tree(x, t, 0) for t in self.trees])
            pl = pl.mean()

            pl_vector.append(pl)

        pl_vector = np.array(pl_vector).reshape(-1, 1)

        return pl_vector

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        input:
            X : dataset
        compute the anomaly score for each x_i observation
        """
        return 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        input:
            scores : return values of anomaly_score function
            threshold
        return an array of the predictions based on threshold.
        """

        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions


class IsolationTree:
    def __init__(self, height_limit, current_height):
        """
        input:
            height_limit : maximum height of the tree
            current_height : current height of the tree
        """
        self.height_limit = height_limit
        self.current_height = current_height
        self.split_by = None    # feature we will split by.
        self.split_value = None # feature's values we will split by.
        self.right = None       # right subtree
        self.left = None        # left subtree
        self.size = 0
        self.exnodes = 0
        self.n_nodes = 1

    def fit(self, X:np.ndarray):
        """
        input:
            X : dataset
        fit function
        """

        if len(X) <= 1 or self.current_height >= self.height_limit:
            self.exnodes = 1
            self.size = X.shape[0]

            return self

        split_by = random.choice(np.arange(X.shape[1]))
        X_col = X[:, split_by]
        min_x = X_col.min()
        max_x = X_col.max()

        if min_x == max_x:
            self.exnodes = 1
            self.size = len(X)

            return self

        else:

            split_value = min_x + random.betavariate(0.5, 0.5) * (max_x - min_x)

            w = np.where(X_col < split_value, True, False)
            del X_col

            self.size = X.shape[0]
            self.split_by = split_by
            self.split_value = split_value

            self.left = IsolationTree(self.height_limit, self.current_height + 1).fit(X[w])
            self.right = IsolationTree(self.height_limit, self.current_height + 1).fit(X[~w])
            self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self



def find_recall_threshold(y, scores, desired_recall):
    """
    Start at score threshold 1.0 and work down until we hit desired recall.
    """
    threshold = 1

    while threshold > 0:
        y_pred = [1 if p[0] >= threshold else 0 for p in scores]
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        recall = tp / (tp + fn)
        if recall >= desired_recall:
            return threshold

        threshold = threshold - 0.001

    return threshold


def c(n):
    """
    to normalize path_length we need c(n)
    """
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0

def path_length_tree(x, t,e):
    """
    give path length of of data instance x in iso_tree t
    """
    e = e
    if t.exnodes == 1:
        e = e+ c(t.size)    # normlization
        return e
    else:
        a = t.split_by
        if x[a] < t.split_value :
            return path_length_tree(x, t.left, e+1)

        if x[a] >= t.split_value :
            return path_length_tree(x, t.right, e+1)
