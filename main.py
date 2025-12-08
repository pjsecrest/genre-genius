import numpy as np
import pandas as pd

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
            feature_cols = []
           
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
            track_cols = []
            
    
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
    
    print(f'X_train: \n {X_train.head()}')
    print(f'y_train: \n {y_train.head()}')


    
def main():
    urls = ['./fma_metadata/features.csv', './fma_metadata/tracks.csv']
    
    features, labels, tracks = transformData(urls) 

    # print(features.head())
    # print(labels.head())
    # print(tracks.head())
    
    loadData(features, labels, tracks)

    
if __name__ == "__main__":
    main()
