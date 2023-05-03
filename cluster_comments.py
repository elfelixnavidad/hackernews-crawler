#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
import numpy as np
import hdbscan
import joblib
import openai
import os
import sys
from itertools import cycle
from pathlib import Path

from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')

import animate

# Set up the OpenAI API client
openai.api_key = os.environ['OPENAI_API_KEY']

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

    df.to_pickle(cluster_file)
    print(f'Done! Saved clusters to {cluster_file}\n')

    return cluster_file

def generate_cluster_label(corpus):
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can read a large body of text and give me a topic category in under 2 words."},
            {"role": "user", "content": corpus},
        ]
    )
    return completion['choices'][0]['message']['content']

def label_clusters(cluster_file):
    df = pd.read_pickle(cluster_file)
    df['cluster_label'] = ''

    cluster_dict = {}
    cluster_id_list = np.sort(df.query('cluster_id >= 0')['cluster_id'].unique())
    
    for c in cluster_id_list:
        filtered_df = df.query('cluster_id == @c')['text']
        max_len = 5 if len(filtered_df) > 5 else len(filtered_df)        
        corpus = ' '.join(filtered_df.iloc[0:max_len])
        
        cluster_dict[c] = generate_cluster_label(corpus)

    df['cluster_label'] = df['cluster_id'].apply(lambda x: cluster_dict[x] if x >= 0 else 'n/a')
    df.to_pickle(cluster_file)

def apply_pca(cluster_file):
    df = pd.read_pickle(cluster_file)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(np.vstack(df['encoding'].to_numpy()))

    # Assign PCA components to separate columns
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]
    df['pca_3'] = pca_result[:, 2]

    df.to_pickle(cluster_file)

    return cluster_file
    
def plot_clusters(cluster_file, show=True):
    df = pd.read_pickle(cluster_file)
    cluster_id_list = np.sort(df.query('cluster_id >= 0')['cluster_id'].unique())
    colors = cycle(cm.tab10.colors)

    fig = plt.figure(figsize=(12, 12), dpi=150)
    ax = fig.add_subplot(projection='3d')

    for c in cluster_id_list:
        color = next(colors)
        cluster_df = df.query('cluster_id == @c')
        cluster_label = cluster_df['cluster_label'].iloc[0]
        
        ax.scatter(cluster_df['pca_1'], cluster_df['pca_2'], cluster_df['pca_3'], color=color, label=cluster_label)        

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    plt.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)
    plt.tight_layout()
    
    if show:
        plt.show()
        return None    
    else:
        plt.axis('off')
        return(fig, ax)

def plot_bar_chart(cluster_file):    
    grouped_df = pd.read_pickle(cluster_file).query('cluster_id >= 0').groupby(by=['cluster_id', 'cluster_label'], as_index=False).agg({'id':'nunique'}).rename(columns={'id':'comments'}).sort_values(by='comments', ascending=False)
    grouped_df['comments'] = (100 * grouped_df['comments'] / grouped_df['comments'].sum()).round(0)

    (fig, ax) = plt.subplots(1, 1, figsize=(12, 12), dpi=125)

    ax.set_ylabel('% of Comments')
    plt.bar(grouped_df['cluster_label'], grouped_df['comments'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./media/barchart_hackernews.png')
    plt.show()
    
def main():
    Path(f'./media').mkdir(parents=True, exist_ok=True)    
    # snapshot_file = './snapshots/hackernews_1681951834.csv'

    # encoding_file = encode_comments(snapshot_file)
    # joblib_file = train_cluster_model(encoding_file)
    # cluster_file = assign_clusters(joblib_file, encoding_file)    
    # label_clusters(cluster_file)
    # apply_pca('./snapshots/hackernews_1681951834_clusters.pkl')
    
    # (fig, ax) = plot_clusters('./snapshots/hackernews_1681951834_clusters.pkl', show=False)
    # angles = np.linspace(0,360,60*10)[:-1] # A list of 20 angles between 0 and 360
    # rotanimate(ax, angles, './media/comment_clusters.mp4', fps=60, bitrate=2000, width=12, height=8)

    plot_bar_chart('./snapshots/hackernews_1681951834_clusters.pkl')
    
if __name__ == '__main__':
    main()
