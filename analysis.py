#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

import etl
import animate

import os
from pathlib import Path
from itertools import cycle

def main():
    Path('./plots').mkdir(parents=True, exist_ok=True)
    
    (comments_df, comments_key) = etl.get_latest_file(prefix='clusters')
    comments_df = comments_df.query('type == \'comment\'')

    comments_key = int(comments_key.replace('clusters/', '').replace('.pkl', ''))
    
    total_comment_count = comments_df['id'].nunique()
    clustered_comment_count = comments_df.query('cluster_id >= 0')['id'].nunique()
    coverage_pct = np.round(100 * clustered_comment_count / total_comment_count, 1)
    
    print(f'There are {total_comment_count} total HackerNews comments in this dataset. We were able to cluster {coverage_pct}% of those comments.')
    
    grouped_df = comments_df\
        .query('cluster_id >= 0')\
        .groupby(by=['cluster_id', 'cluster_label'], as_index=False)\
        .agg({'id':'nunique'})\
        .rename(columns={'id':'comments'})\
        .sort_values(by='comments', ascending=False)
    
    grouped_df['comment_pct'] = (100 * grouped_df['comments'] / grouped_df['comments'].sum()).round(0)

    print(grouped_df['cluster_id'].nunique())
    # (fig, ax) = plt.subplots(1, 1, figsize=(12, 12), dpi=125)    
    # ax.set_ylabel('% of Comments')
    # plt.bar(grouped_df['cluster_label'], grouped_df['comment_pct'])
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.savefig(f'./plots/barchart_pct_{comments_key}.png')
    # plt.close()

    # (fig, ax) = plt.subplots(1, 1, figsize=(12, 12), dpi=125)    
    # ax.set_ylabel('Comments')
    # plt.bar(grouped_df['cluster_label'], grouped_df['comments'])
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.savefig(f'./plots/barchart_{comments_key}.png')
    # plt.close()

    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(np.vstack(comments_df['encoding'].to_numpy()))

    # comments_df['pca_1'] = pca_result[:, 0]
    # comments_df['pca_2'] = pca_result[:, 1]
    # comments_df['pca_3'] = pca_result[:, 2]

    # cluster_id_list = np.sort(comments_df.query('cluster_id >= 0')['cluster_id'].unique())
    # colors = cycle(cm.tab10.colors)

    # fig = plt.figure(figsize=(12, 12), dpi=150)
    # ax = fig.add_subplot(projection='3d')

    # for c in cluster_id_list:
    #     color = next(colors)
    #     cluster_df = comments_df.query('cluster_id == @c')
    #     cluster_label = cluster_df['cluster_label'].iloc[0]        
    #     ax.scatter(cluster_df['pca_1'], cluster_df['pca_2'], cluster_df['pca_3'], color=color, label=cluster_label)        

    # ax.set_xlabel('PCA 1')
    # ax.set_ylabel('PCA 2')
    # ax.set_zlabel('PCA 3')
    # plt.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)
    # plt.tight_layout()
    # plt.axis('off')    

    # angles = np.linspace(0,360,60*10)[:-1]
    # animate.rotanimate(ax, angles, f'./plots/comments_{comments_key}.mp4', fps=60, bitrate=2000, width=12, height=8)

if __name__ == '__main__':
    main()
