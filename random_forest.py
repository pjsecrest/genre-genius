import numpy as np
import pandas as pd
from tree import DecisionTree
from bagging_ensemble import BaggingEnsemble
from sklearn.tree import DecisionTreeClassifier


# CODE FOR RANDOM FOREST
def random_selection(input_dim, output_dim):
    """ Randomly sample output_dim indices in range [0, input_dim-1]
    Returns:
        indices array of size (output_dim,)
    """
    assert input_dim >= output_dim
    # Workspace 2.7
    selected_features = None
    #BEGIN
    # idxs = np.random.randint(0, X_train.shape[0], int(np.floor(self.sample_ratio*X_train.shape[0])))
    selected_features = np.random.choice(input_dim, size=output_dim, replace=False)
    #END
    return selected_features

class RandomForest(BaggingEnsemble):

    def __init__(self, n_estimators, sample_ratio=1.0, features_ratio=1.0):
        super(RandomForest, self).__init__(n_estimators, sample_ratio)
        self.features_ratio = features_ratio
        self.estimators = []  # to store the estimator
        self.selections = []  # to store the feature indices used by each estimator

    def sample_data(self, X_train, y_train):

        input_dim = None
        output_dim = None
        indices = None
        # Workspace 2.8
        #BEGIN

        input_dim = X_train.shape[1]
        output_dim = int(self.features_ratio * input_dim)
        indices = np.random.choice(X_train.shape[0], size=int(self.sample_ratio * X_train.shape[0]), replace=True)
 
        #END
        selected_features = random_selection(input_dim, output_dim)
        return X_train[indices][:, selected_features], y_train[indices], selected_features

    def fit(self, X_train, y_train):
        print('fitting')
        # np.random.seed(42)  # keep to have consistent results across run, you can change the value
        self.estimators = []  # to store the estimator
        self.selections = []

        for _ in range(self.n_estimators):
            # Workspace 2.9
            # TODO: sample data with random subset of rows and features using sample_data
            # Hint: keep track of the projections to use in predict
            #BEGIN
        
            x_sample, y_sample, selections = self.sample_data(X_train, y_train)
            
            # dt = DecisionTreeClassifier()
            dt = DecisionTreeClassifier()
            dt.fit(x_sample, y_sample)

            self.estimators.append(dt)
            self.selections.append(selections)
            #END

    def predict(self, X_test):
        predicted_proba = 0
        answer = 0
        # Workspace 2.10
        # TODO: compute cumulative sum of predict proba from estimators and return the labels with highest likelihood
        #BEGIN
    
        for i, e in enumerate(self.estimators):
            features = self.selections[i]
            test_set = X_test[:, features]
            
            proba = e.predict_proba(test_set)
            predicted_proba += proba

        answer = np.argmax(predicted_proba, axis=1)
        
        #END
        return answer