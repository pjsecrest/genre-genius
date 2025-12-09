import numpy as np
import pandas as pd
from tree import DecisionTree

def get_weak_learner():
    """Return a new instance of out chosen weak learner"""
    return DecisionTree(max_depth=2, min_samples_leaf=0.1)

class BaggingEnsemble(object):

    def __init__(self, n_estimators, sample_ratio=1.0):
        """
        Initialize BaggingEnsemble
        :param n_estimators: number of estimators/weak learner to use
        :param sample_ratio: ratio of the training data to sample
        """
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.estimators = []  # List used in fit method to store the trained estimators

    def sample_data(self, X_train, y_train):
        X_sample, y_sample = None, None
        # Workspace 2.4
        # TODO: sample random subset of size sample_ratio * len(X_train), sampling is with replacement (iid)
        #BEGIN

        idxs = np.random.randint(0, X_train.shape[0], int(np.floor(self.sample_ratio*X_train.shape[0])))

        X_sample = X_train[idxs, :]
        y_sample = y_train[idxs]
        
        #END
        return X_sample, y_sample

    def fit(self, X_train, y_train):
        """
        Train the different estimators on sampled data using provided training samples
        :param X_train: training samples, shape (num_samples, num_features)
        :param y_train: training labels, shape (num_samples)
        :return: self
        """
        np.random.seed(42)  # Keep it to get consistent results across runs, you can change the seed value
        self.estimators = []

        for _ in range(self.n_estimators):
            # Workspace 2.5
            #BEGIN

            X_sample, y_sample = self.sample_data(X_train, y_train)
            
            estimator = get_weak_learner()
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)
            
            #END
        return self

    def predict(self, X_test):
        """
        Predict the labels of test samples
        :param X_test: array of shape (num_points, num_features)
        :return: 1-d array of shape (num_points)
        """
        predicted_proba = 0
        answer = 0
        # Workspace 2.6
        # TODO: go through the trained estimators and accumulate their predicted_proba to get the mostly likely label
        #BEGIN
        probas = []
        for e in self.estimators:
            proba = e.predict_proba(X_test)
            probas.append(proba)
        
        avg = np.mean(probas, axis=0)
        answer = np.argmax(avg, axis = 1)
        
        #END
        return answer

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
