#!/usr/bin/env python3
import json
import time
import re
import os
import datetime
import html
import pickle
import io

from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import hdbscan
import openai
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from sentence_transformers import SentenceTransformer, util

SEP = '\t'
model = SentenceTransformer('all-mpnet-base-v2')
openai.api_key = os.environ['OPENAI_API_KEY']

STORIES_DIR = 'stories'
EMBEDDINGS_DIR = 'embeddings'

def call_hackernews_api(endpoint):
    """
    Call the HackerNews web API via the v0 endpoint.
    """
    url = f'https://hacker-news.firebaseio.com/v0/{endpoint}'
    data = None
    
    try:
        with urlopen(url) as response:
            body = response.read()
    except HTTPError as e:
        print(f"HTTPError = {e.code}")
    except URLError as e:
        print(f"URLError = {e.reason}")
    except Exception as e:
        import traceback
        print(f"Generic exception: {traceback.format_exc()}")
    else:
        data = json.loads(body)

    return data

def create_row(id_dict, parent_id):
    """
    Sample schema
    | post_timestamp |  post_id | parent_id | post_type | post_by     | post_text                                                                                                                                                                                             |
    |----------------+----------+-----------+-----------+-------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    |     1681917047 | 35629127 |           | story     | davidbarker | StableLM: A new open-source language model{SEP}https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models{SEP}1074                                            |
    |     1681924196 | 35630664 |  35629127 | comment   | dang        | <a href="https://github.com/Stability-AI/StableLM">https://github.com/Stability-AI/StableLM</a>                                                                                                       |
    |     1681941664 | 35633859 |  35633163 | comment   | jvm         | Doesn't make much sense to compare a model that's not fine tuned to flan models that are fine tuned. Makes more sense to compare to something like T5 base where it's probably a lot more comparable. |
    """
    post_timestamp = ''
    post_id = ''
    post_type = ''
    post_by = ''
    post_text = ''    

    if (not id_dict) or ('deleted' in id_dict.keys() and id_dict['deleted']) or ('dead' in id_dict.keys() and id_dict['dead']):
        return f'{post_timestamp}{SEP}{post_id}{SEP}{parent_id}{SEP}{post_type}{SEP}{post_by}{SEP}{post_text}'
        
    if 'time' in id_dict.keys():
        post_timestamp = id_dict['time']
    
    if 'id' in id_dict.keys():
        post_id = id_dict['id']
    
    if 'type' in id_dict.keys():
        post_type = id_dict['type']
    
    if 'by' in id_dict.keys():
        post_by = id_dict['by']

    if post_type == 'comment':
        if 'text' in id_dict.keys():
            # re.sub(r'[^a-zA-Z0-9 .!?,]', '', id_dict['text'])
            post_text = html.unescape(id_dict['text']) 
    elif post_type == 'story':
        if 'title' in id_dict.keys():
            # re.sub(r'[^a-zA-Z0-9 .!?,]', '', )
            post_text += id_dict['title']
        if 'url' in id_dict.keys():
            post_text += '|' + id_dict['url']
        if 'score' in id_dict.keys():
            post_text += '|' + str(id_dict['score'])
    
    return f'{post_timestamp}{SEP}{post_id}{SEP}{parent_id}{SEP}{post_type}{SEP}{post_by}{SEP}{post_text}'

def traverse_comment(id_dict, parent_):
    """
    Recurse thru comment thread.
    """
    data = create_row(id_dict, parent_)
    
    if 'kids' not in id_dict.keys():
        return data + '\n'
    else:
        for k in id_dict['kids']:
            data += '\n' + traverse_comment(call_hackernews_api(f'item/{k}.json'), id_dict['id'])

    return data

def upload_bytes_to_s3(body, key):
    session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )

    s3 = session.resource('s3')
    object = s3.Object(os.environ['AWS_HACKERNEWS_BUCKET'], key)
    object.put(Body=body)    

def download_bytes_from_s3(key):
    session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )

    s3_client = session.client('s3')
    s3_response_object = s3_client.get_object(Bucket=os.environ['AWS_HACKERNEWS_BUCKET'], Key=key)
    return s3_response_object['Body'].read()

def list_files_in_bucket(prefix):
    session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )

    client = session.client('s3')
    paginator = client.get_paginator('list_objects')
    page_iterator = paginator.paginate(Bucket=os.environ['AWS_HACKERNEWS_BUCKET'], Prefix=f'{prefix}/')    
    page_list = [p['Contents'] for p in page_iterator]
    content_list = []
    
    for p in page_list:
        for c in p:
            content_list.append(c['Key'])
        
    return content_list


def save_story(story_id):
    """
    Save HackerNews story and all associated comments.
    """
    story_data = traverse_comment(id_dict=call_hackernews_api(f'item/{story_id}.json'), parent_='')
    write_ts = int(time.time())
    upload_bytes_to_s3(body=story_data, key=f'{STORIES_DIR}/{story_id}/{write_ts}.csv')
    
def crawl():
    """
    Look thru the last 500 HackerNews stories and save a snapshot of stories and associated comments.
    """
    new_stories_list = call_hackernews_api('newstories.json')
    story_count = 1
    for s in new_stories_list:
        print(f'[{story_count}/500]: Saving {s}')
        save_story(s)
        story_count += 1

def create_crawl_snapshot():
    """
    Compile all CSVs from base_directory and flatten them into one snapshot.
    """
    write_ts = int(time.time())
    encoded_comments_key = f'{EMBEDDINGS_DIR}/{write_ts}.pkl'
    
    df_list = []
    columns = ['timestamp', 'id', 'parent_id', 'type', 'by', 'text']
    
    for f in list_files_in_bucket(prefix=STORIES_DIR):
        if '.csv' in f:
            (prefix, story_id, write_ts) = f.replace('.csv', '').split('/')

            df = pd.read_csv(io.StringIO(download_bytes_from_s3(key=f).decode('utf-8')), names=columns, sep=SEP)
            df['story_id'] = story_id
            df['write_ts'] = write_ts
            df_list.append(df)

    merged_df = pd.concat(df_list).dropna(subset=['id']).drop_duplicates(subset=['id', 'text'])
    merged_df = merged_df[['timestamp', 'story_id', 'id', 'parent_id', 'type', 'by', 'text', 'write_ts']]
    merged_df['encoding'] = merged_df['text'].apply(lambda x: model.encode(x))

    upload_bytes_to_s3(body=pickle.dumps(merged_df), key=encoded_comments_key)

    return encoded_comments_key
    
def get_latest_file(prefix):
    sorted_list = np.sort(list_files_in_bucket(prefix))
    key = sorted_list[-1]
    
    return (pickle.loads(download_bytes_from_s3(key)), key)

def generate_cluster_label(corpus):
    role = """
    You are topicGPT, you can read a large body of text and then create a topic category that describes the body of text. 
    For example if you read comments about ways to fix broken engines, smog checks, and check engine lights you would reply with auto maintenance as the topic category.
    You will limit yourself to 4 words or less per topic category. After you create your topic category double check you meet the word limit requirement before giving your final answer.
    """
    
    completion = openai.ChatCompletion.create(
        # model='gpt-4',
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": corpus},
        ]
    )
    return completion['choices'][0]['message']['content']

def cluster_comments():
    print(f'Creating HDBSCAN model')
    (df, key) = get_latest_file(prefix='embeddings')

    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(np.vstack(df['encoding'].to_numpy()))

    print('Assigning cluster IDs')
    df['cluster_id'] = clusterer.labels_    
    df['cluster_label'] = ''

    cluster_id_list = np.sort(df.query('cluster_id >= 0')['cluster_id'].unique())
    cluster_dict = {}

    print('Creating topic categories and assigning cluster labels')
    for c in cluster_id_list:
        filtered_df = df.query('cluster_id == @c')['text']
        max_len = 5 if len(filtered_df) > 5 else len(filtered_df)        
        corpus = ' '.join(filtered_df.iloc[0:max_len])
        
        cluster_dict[c] = generate_cluster_label(corpus)

    df['cluster_label'] = df['cluster_id'].apply(lambda x: cluster_dict[x] if x >= 0 else 'n/a')
    
    print('Uploading to S3')
    upload_bytes_to_s3(body=pickle.dumps(df), key=key.replace('embeddings', 'clusters'))
    
def crawl_to_s3():
    crawl()
    print('Creating embeddings...')
    create_crawl_snapshot()
    print('Creating cluster model...')
    cluster_comments()
    
if __name__ == '__main__':
    crawl_to_s3()
