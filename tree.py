import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score


def compute_label(node_labels):

    # Workspace 1.1
    # TODO: Return the label that should be assigned to the leaf node
    # In case of multiple possible labels, choose the one with the highest value
    # Make no assumptions about the number of class labels
    label = None
    #BEGIN
    unique_labels, counts = np.unique(node_labels, return_counts=True)
    if np.all(counts[0] == counts):
        label = max(unique_labels)
    else:
        max_idx = np.argmax(counts)
        label = unique_labels[max_idx]
    #END
    return label



class LeafNode:
    def __init__(self, node_labels):
        """ Initialize the leaf node
        Args:
            y: 1-d array containing labels, of shape (num_points,)
        """
        self.label = compute_label(node_labels)

    @staticmethod
    def is_terminal():
        return True

    def predict(self, X):
        return self.label * np.ones(X.shape[0])
    

def gini(y):
    """
    Args:
        y: 1-d array contains labels, of shape (num_points,)
    Returns: float, gini impurity of the values in y
    """
    gini = 0
    # Workspace 1.2
    # TODO: Compute the gini impurity of the labels
    #BEGIN
    print(y)
    labels, counts = np.unique(y, return_counts=True)
    gini = 1 - np.sum([(count / len(y))**2 for count in counts])
    #END
    return gini

def impurity_reduction(y, left_indices, right_indices):
    """
    Args:
        y: all labels
        left_indices: the indices of the elements of y that belong to the left child
        right_indices: the indices of the elements of y that belong to the right child
    Returns: impurity reduction of the split
    """
    reduction = 0
    # Workspace 1.4
    #BEGIN
    left_split = y.iloc[left_indices]
    right_split = y.iloc[right_indices]
    reduction = gini(y) - ((len(left_split) / len(y) * gini(left_split)) + (len(right_split) / len(y) * gini(right_split)))
    #END
    return reduction


def split_values(feature_values):
    """ Helper function to return the split values. if feature consists of the values f1 < f2 < f3 then
    this returns [(f2 + f1)/2, (f3 + f2)/2]
    Args:
        feature_values: feature_values: 1-d array of shape (num_points)
    Returns:  array of shape (max(m-1, 1),) where m is the number of unique values in feature_values
    """
    unique_values = np.unique(feature_values)
    if unique_values.shape[0] == 1:
        return unique_values
    return (unique_values[1:] + unique_values[:-1]) / 2


def best_split(X, y):
    """ Find the feature id, threshold, indices, and reduction for the best split
    Args:
        X: features array, shape (num_samples, num_features)
        y: labels of instances in X, shape (num_samples)
    Returns: the best split related information.
    """

    best_feature_id, best_threshold, best_left_indices, best_right_indices = None, None, None, None
    best_reduction = -np.inf

    # Workspace 1.5
    # TODO: Complete the function as detailed in the question and return description
    # NOTE: See specification in Q1.6:
    #       if feature_value == threshold, it should end up in the **left** child.
    for feature_id in range(X.shape[1]):
        print(f" ---------- feature: {feature_id} ----------")
        for threshold in split_values(X.iloc[:, feature_id]):
        #BEGIN

            left_split_indices = np.where(X.iloc[:, feature_id] < threshold)[0]
            right_split_indices = np.where(X.iloc[:, feature_id] >= threshold)[0]

            if len(left_split_indices) == 0 or len(right_split_indices) == 0:
                continue 

            reduction = impurity_reduction(y, left_split_indices, right_split_indices)

            if reduction > best_reduction:
                best_reduction = reduction
                best_feature_id = feature_id
                best_threshold = threshold
                best_left_indices = left_split_indices
                best_right_indices = right_split_indices
        #END
    return best_feature_id, best_threshold, best_left_indices, best_right_indices, best_reduction


class DecisionNode:
    def __init__(self, feature_id, threshold, left_child, right_child):
        self.feature_id = feature_id
        self.threshold = threshold
        self.left = left_child
        self.right = right_child

    @staticmethod
    def is_terminal():
        return False

    def add_importance(self, importances: dict, X, y):
        # Workspace 1.10
        # Bonus question
        # Note that dictionaries are passed by reference and not by value
        #BEGIN

        # this was extra credit in the homework
        
        #END
        return importances
    
    def predict(self, X):
        y_pred = np.zeros((X.shape[0]))
        left_indices = np.where(X[:, self.feature_id] <= self.threshold)[0]
        right_indices = np.where(X[:, self.feature_id] > self.threshold)[0]
        y_pred[left_indices] = self.left.predict(X[left_indices])
        y_pred[right_indices] = self.right.predict(X[right_indices])
        return y_pred
    

def build_tree(X, y, depth=-1, min_samples_split=2):
        if depth == 0 or len(y) < min_samples_split:
            # we reached the maximum depth or we don't have more than the minimum number of samples in the leaf
            tree = LeafNode(y)
        else:
            # Get the feature, threshold and information_gain of the best split
            feature_id, threshold, left_indices, right_indices, reduction = best_split(X, y)
            # reduction = 0 occurs when the labels have the same distribution in the child nodes
            # which means that the entropy of the children is the same as the parent's so we don't need to split
            # Workspace 1.6
            # TODO: if needed, create the left and right child nodes with depth - 1, return the decision node
            #BEGIN
            if reduction <= 0:
                tree = LeafNode(y)
            else:

                left_child = build_tree(X[left_indices], y[left_indices], depth - 1, min_samples_split)
                right_child = build_tree(X[right_indices], y[right_indices], depth - 1, min_samples_split)
        
                tree = DecisionNode(feature_id, threshold, left_child, right_child)
            #END
        return tree


class DecisionTree:

    def __init__(self, max_depth=-1, min_samples_split=2):
        """ Initialize the decision tree
        Args:
            max_depth: maximum depth of the tree
            min_samples_split: minimum number of samples required for a split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.num_features = None


    def fit(self, X, y):
        """
        Args:
            X: Training samples
            y: training labels
        Return:
             trained decision tree
        """
        self.tree = build_tree(X, y, self.max_depth, self.min_samples_split)
        return self

    def predict(self, X):
        """
        Loops through rows of X and predicts the labels one row at a time
        """
        return self.tree.predict(X)

    def feature_importance(self, X, y):
        """ Compute the importance of each feature in the decision tree
         Only relevant to the bonus question
        """
        feat_importance = {k:0 for k in range(X.shape[1])}
        if not self.tree.is_terminal():
            self.tree.add_importance(feat_importance, X, y)
        feat_importance = {k: v/sum(feat_importance.values()) for k,v in feat_importance.items()}
        return feat_importance

    def score(self, X, y):
        """ Return the mean accuracy on the given test data and labels.
        Args:
            X: Test samples, shape (num_points, num_features)
            y: true labels for X, shape (num_points,)
        Return:
            mean accuracy
        """
        accuracy = 0
        # Workspace 1.7
        #BEGIN
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y) 
        #END
        return accuracy



