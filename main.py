import numpy as np
import pandas as pd
import tree
from random_forest import RandomForest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# format data headers and return features, labels, and other track metadata
def transformData(urls):

    features = pd.read_csv('./fma_metadata/features.csv', header=[0, 1, 2], low_memory=False)
    tracks = pd.read_csv('./fma_metadata/tracks.csv', header=[0, 1], low_memory=False)

    # consolidate features headers
    # Step 2: Drop the extra row containing "track_id" in first column
    features = features.iloc[1:].reset_index(drop=True)
    tracks = tracks.iloc[1:].reset_index(drop=True)
    
    # # Step 3: Fix the first column name back to "track_id"
    features = features.rename(columns={features.columns[0]: 'track_id'})

    # # Step 4: Flatten multi-index columns (except first column)
    feature_cols = []
    for col in features.columns:
        if isinstance(col, tuple):
            # first column ("track_id") tuple looks like ("track_id", "", "") → handle separately
            if col[0] == 'track_id':
                feature_cols.append('track_id')
            else:
                # combine non-empty parts
                parts = [str(x) for x in col if x not in ('', 'nan', None)]
                feature_cols.append('_'.join(parts))
        else:
            feature_cols.append(col)
           
    track_cols = []
    for col in tracks.columns:
        if isinstance(col, tuple):
            # first column ("track_id") tuple looks like ("track_id", "", "") → handle separately
            if col[0] == 'track_id':
                track_cols.append('track_id')
            else:
                # combine non-empty parts
                parts = [str(x) for x in col if x not in ('', 'nan', None)]
                track_cols.append('_'.join(parts))
        else:
            track_cols.append(col)
            
    
    # set columns and reindex with track_id
    features.columns = feature_cols
    features = features.rename({'feature_statistics_number': 'track_id'}, axis=1)
    features = features.set_index('track_id')
    
    tracks.columns = track_cols
    tracks = tracks.rename({'Unnamed: 0_level_0_Unnamed: 0_level_1': 'track_id'}, axis=1)
    # tracks = tracks.set_index('track_id')
    
    # extract track_id and genre labels
    # TODO: Potentially need to extract genre id -> labels, track_genres and track_genres_all are lists of genre_ids
    labels = tracks[['track_id', 'track_genre_top', 'track_genres', 'track_genres_all']]
    labels = labels.set_index('track_id')
    
    tracks = tracks.set_index('track_id')
    # print(tracks.head()) 
    # print(features.head())
    # print(labels.head())
    
    # transform labels to integer ids 
    # genres = pd.read_csv('./fma_metadata/genres.csv')
    genres = pd.read_csv("./fma_metadata/genres.csv")
    genre_title_id_map = dict(zip(genres['title'], genres['genre_id']))

    labels['track_genre_top_id'] = labels['track_genre_top'].map(genre_title_id_map)
    
    return features, labels, tracks
    
# get training, test, and validation splits
def loadData(features, labels, tracks):
    # merge
    data = features.join(labels, how='inner')
    data = data.join(tracks[['set_split']], how='inner')
    
    # # split
    X = data[features.columns]
    y = data['track_genre_top_id']

    train_idx = data['set_split'] == 'training'
    val_idx = data['set_split'] == 'validation'
    test_idx = data['set_split'] == 'test'
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
# evaluate the predictions against truth labels
def evaluatePredictions(y_pred, y_true, k=3):

    f1 = 0
    accuracy_score = np.mean(y_pred == y_true)
    top_k_accuracy = 0
    # confusion_matrix = None
    
    # confusion
    
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    
    # calc F1 
    # tp, tn, fp, fn = 0, 0, 0, 0
    
    # tp = np.count_nonzero(y_pred == y_true)
    # tn = np.count_nonzero()
    
    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)
    # f1 = (2*precision*recall)/(precision+recall)
    # print(f'F1 Score: {f1}')
    
    return
    
def main():
    urls = ['./fma_metadata/features.csv', './fma_metadata/tracks.csv']
    
    features, labels, tracks = features, labels, tracks = transformData(urls)
    X_train, X_val, X_test, y_train, y_val, y_test = loadData(features, labels, tracks)
    
    
    # InitTree = tree.DecisionTree(max_depth = 2, min_samples_split=2)
    # InitTree.fit(X_train, y_train)

    # print("Built and Fitted Tree")

    # print( f"Score on predictions (Tree):  {InitTree.score(X_test.to_numpy(), y_test.to_numpy())}")
    # X_train = X_train.head(10).to_numpy()
    # y_train = y_train.head(10).to_numpy()
    # X_test = X_test.head(10).to_numpy()
    
    combined = pd.concat([X_train, y_train], axis=1)
    combined = combined.dropna()
    
    y_train = combined.iloc[:, -1]
    X_train = combined.iloc[:, :-1]
    
    combined_test = pd.concat([X_train, y_train], axis=1)
    combined_test = combined.dropna()
    
    y_test = combined_test.iloc[:, -1]
    X_test = combined_test.iloc[:, :-1]
    
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    # TODO: add eval stats(F1, confusion matrix? will be massive, top K accuracy (predicting top k genres rather than top 1))
    # TODO: hyperparameter tuning, must find optimal: n_estimators, sample_ratio, features_ratio, max_depth, max_features

    rf = RandomForest(1)
    
    
    
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)
    
    print(rf_predictions.shape)
    print(y_test.shape)
    
    # RF evaluation
    evaluatePredictions(rf_predictions, y_test)
    print(f'Random Forest Predictions (top 1): {rf_predictions}')
    

if __name__ == "__main__":
    main()
