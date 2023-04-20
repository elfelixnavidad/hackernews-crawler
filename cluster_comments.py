#!/usr/bin/env python3
import pandas as pd
import numpy as np
import hdbscan
import joblib

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')

def encode_comments(comment_file):
    print(f'Encoding {comment_file}')

    pickle_file = comment_file.replace('.csv', '_encoding.pkl')
    df = pd.read_csv(comment_file, sep='\t').query('type == \'comment\'')
    df['encoding'] = df['text'].apply(lambda x: model.encode(x))
    df.to_pickle(pickle_file)
    
    print(f'Done! Saved to {pickle_file}\n')

    return pickle_file

def train_cluster_model(encoding_file):
    print(f'Clustering {encoding_file}')    
    df = pd.read_pickle(encoding_file)    
    
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(np.vstack(df['encoding'].to_numpy()))

    joblib_file = encoding_file.replace('_encoding.pkl', '_cluster_model.joblib')
    joblib.dump(clusterer, joblib_file)
    
    print(f'Done! Saved to {joblib_file}\n')

    return joblib_file

def assign_clusters(joblib_file, encoding_file):
    print(f'Assigning clusters based on {joblib_file} model and {encoding_file} encodings')
    cluster_file = encoding_file.replace('encoding', 'clusters')
    model = joblib.load(joblib_file)

    df = pd.read_pickle(encoding_file)
    df['cluster_id'] = model.labels_

    df.drop(columns=['encoding']).to_pickle(cluster_file)
    print(f'Done! Saved clusters to {cluster_file}\n')
    
def main():
    snapshot_file = '/home/felix/Documents/Python/hackernews-crawler/snapshots/hackernews_1681951834.csv'
    
    encoding_file = encode_comments(snapshot_file)
    joblib_file = train_cluster_model(encoding_file)
    assign_clusters(joblib_file, encoding_file)    
    
if __name__ == '__main__':
    main()
