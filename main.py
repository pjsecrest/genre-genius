import numpy as np
import pandas as pd

def loadFeatureData(urls):

    X_data = pd.read_csv('./fma_metadata/features.csv')
    # y_data = pd.read_csv('./fma_metadata/genres.csv')
    tracks = pd.read_csv('./fma_metadata/tracks.csv')
    
    print(X_data.head())
    print(tracks.head())
    
    
def main():
    urls = ['./fma_metadata/features.csv', './fma_metadata/tracks.csv']

    loadFeatureData(urls)    
    
if __name__ == "__main__":
    main()
