import ast
import numpy as np
import pandas as pd
import tree
from random_forest import RandomForest

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, top_k_accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns

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
    # print(features)
    
    # flatten data headers
    flat_feature_cols = []
    for col in features.columns:
        if col[0] == 'feature' and col[1] == 'statistics' and col[2] == 'number':
            flat_feature_cols.append("track_id")
        else:
            parts = [str(x) for x in col if (x != '' and x is not None)]
            flat_feature_cols.append("_".join(parts))

    features.columns = flat_feature_cols
    features = features.set_index('track_id')    
    # features.to_csv('./data/temp_Features.csv')

    flat_track_cols = []
    for col in tracks.columns:
        if col[0] == 'track_id':
            flat_track_cols.append('track_id')
        else:
            parts = [str(x) for x in col if (x != '' and x is not None)]
            flat_track_cols.append("_".join(parts))

    tracks.columns = flat_track_cols

    tracks = tracks.rename({'Unnamed: 0_level_0_Unnamed: 0_level_1':'track_id'}, axis=1)
    tracks = tracks.set_index('track_id')
    
    # tracks.to_csv('./data/flattened_tracks.csv')
    # features.to_csv('./data/flattened_features.csv')
    
    # extract track_id and genre labels
    labels = tracks[['track_genre_top', 'track_genres', 'track_genres_all']]
    # labels = labels.set_index('track_id')
    labels['track_genres_all'] = labels['track_genres_all'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else ([] if pd.isna(x) else x)
    )
    
    # GENRE DEFINITIONS (map ids to one hot encoded indexes)
    genres = pd.read_csv("./fma_metadata/genres.csv")
    
    all_genre_ids = genres['genre_id'].tolist()
    genre_id_idx_map = {id: i for i, id in enumerate(all_genre_ids)}
    
    genre_count = len(all_genre_ids)
    
    # BUILD Y_MULTILABEL
    track_ids = labels.index.tolist()
    track_count = len(track_ids)
    
    y_multilabel = np.zeros((track_count, genre_count), dtype=int)
 
    for row_i, genre_list in enumerate(labels['track_genres_all']):
        for g in genre_list:
            if g in genre_id_idx_map:
                col = genre_id_idx_map[g]
                y_multilabel[row_i, col] = 1
    
    y_multilabel_df = pd.DataFrame(
        y_multilabel,
        index=track_ids,
        columns=all_genre_ids   # columns named by genre_id
    )
    
    # y_multilabel_df.to_csv('./data/y_multilabel_df.csv')
    # transform labels to integer ids
    genre_title_id_map = dict(zip(genres['title'], genres['genre_id']))
    labels['track_genre_top_id'] = labels['track_genre_top'].map(genre_title_id_map)
    
    return features, labels, tracks, y_multilabel_df
# get training, test, and validation splits
def loadData(features, labels, tracks, y_multilabel):
    # merge
    data = features.join(labels, how='inner')
    data = data.join(tracks[['set_split']], how='inner')
    
    # # split
    X = data[features.columns]
    y_top = data['track_genre_top_id'] # single top genre truth
    

    train_idx = data['set_split'] == 'training'
    val_idx = data['set_split'] == 'validation'
    test_idx =  data['set_split'] == 'test'
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    
    y_train_top = y_top[train_idx]
    y_val_top = y_top[val_idx]
    y_test_top = y_top[test_idx]
    
    y_train_multi = pd.DataFrame(y_multilabel[train_idx])
    y_val_multi = pd.DataFrame(y_multilabel[val_idx])
    y_test_multi =pd.DataFrame(y_multilabel[test_idx])
    
    return X_train, X_val, X_test, y_train_top, y_val_top, y_test_top, y_train_multi, y_val_multi, y_test_multi
    
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
    
    features, labels, tracks, y_multilabel = transformData(urls)
    # print(y_multilabel)
    # print(y_multilabel.index)
    X_train, X_val, X_test, y_train_top, y_val_top, y_test_top, y_train_multi, y_val_multi, y_test_multi = loadData(features, labels, tracks, y_multilabel)
    
    
    # After transformData
    # print(y_train.value_counts().sort_values(ascending=False))
    # print(f"\nNumber of unique classes in training: {y_train.nunique()}")
    # print(f"Number of unique classes in test: {y_test.nunique()}")
 
    # Drop NaN values
    # combined = pd.concat([X_train, y_train_top], axis=1)
    # combined = combined.dropna()
    # y_train = combined.iloc[:, -1]
    # X_train = combined.iloc[:, :-1]
    
    # combined_test = pd.concat([X_test, y_test_top], axis=1)
    # combined_test = combined.dropna()
    # y_test = combined_test.iloc[:, -1]
    # X_test = combined_test.iloc[:, :-1]
    
    # convert to np arrays for use in models
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # y_train_top = y_train.to_numpy()
    # y_test_top = y_test.to_numpy()
    
    y_train_multi = y_train_multi.to_numpy()
    y_val_multi = y_val_multi.to_numpy()
    y_test_multi = y_test_multi.to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # TODO: add eval stats(F1, confusion matrix? will be massive, top K accuracy (predicting top k genres rather than top 1))
    # TODO: hyperparameter tuning, must find optimal: n_estimators, sample_ratio, features_ratio, max_depth, max_features

    # Initialize and fit models, then predict
    lr_classifier = LogisticRegression(max_iter=5000, C=0.1)
    # rf_classifier = RandomForestClassifier(n_estimators=5)
    # lgb_classifier = LGBMClassifier(
    #     boosting_type='gbdt',
    #     num_leaves=50,
    #     learning_rate=0.1,
    #     n_estimators=100,
    #     n_jobs=-1
    # )
    
    multi_out_classifier_lr = OneVsRestClassifier(lr_classifier)
    # multi_out_classifier_rf = OneVsRestClassifier(rf_classifier)
    # multi_out_classifier_lgb = OneVsRestClassifier(lgb_classifier, n_jobs=-1)
    # multi_out_classifier_lgb.fit(X_train_scaled, y_test)

    multi_out_classifier_lr.fit(X_train_scaled, y_train_multi)
    multi_lr_prediction = multi_out_classifier_lr.predict(X_test_scaled)
    
    # lr_classifier.fit(X_train_scaled, y_train_top)
    # single_lr_prediction = lr_classifier.predict(X_test_scaled)
    # multi_rf_prediction = multi_out_classifier_rf.predict(X_test_scaled)
    # lgb_prediction = multi_out_classifier_lgb.predict(X_test_scaled)

    # evaluations
    multi_lr_f1 = f1_score(y_test_multi, multi_lr_prediction, average='macro') # TODO change to macro
    multi_lr_score = accuracy_score(y_test_multi, multi_lr_prediction)
    
    # single_lr_f1 = f1_score(y_test_top, single_lr_prediction, average='macro') # TODO change to macro
    # single_lr_score = accuracy_score(y_test_top, single_lr_prediction)

    
    # multi_rf_f1 = f1_score(y_test_multi, multi_rf_prediction, average='macro')
    # multi_rf_score = accuracy_score(y_test_multi, multi_rf_prediction)
    
    # lgb_f1 = f1_score(y_test, lgb_prediction, average='micro')
    # lgb_score = accuracy_score(y_test, lgb_prediction)
    
    # print(f'Multi LGB predictions (16 label): {lgb_prediction}')
    # print(f'Multi XBG F1 (16 label): {lgb_f1}')
    # print(f'Multi XBG Score (16 label): {lgb_score}')
    
    print(f'Multi LR predictions: {multi_lr_prediction}')
    print(f'Multi LR F1: {multi_lr_f1}')
    print(f'Multi LR Score: {multi_lr_score}')
    
    # print(f'Single LR predictions: {single_lr_prediction}')
    # print(f'Single LR F1: {single_lr_f1}')
    # print(f'Single LR Score: {single_lr_score}')
    
    # print(f'Multi RF predictions: {multi_rf_prediction}')
    # print(f'Multi RF F1 (16 label): {multi_rf_f1}')
    # print(f'Multi RF Score (16 label): {multi_rf_score}')
    
    # np.savetxt('./predictions/rf_predictions_n25.csv', multi_rf_prediction)
    # np.savetxt('./predictions/lr_predictions.csv', multi_lr_prediction)    
    # np.savetxt('./predictions/xgb_multi_predictions.csv', lgb_prediction)

if __name__ == "__main__":
    main()