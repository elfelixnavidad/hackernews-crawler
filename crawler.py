#!/usr/bin/env python3
import json
import time
import re
import os
import datetime
import html
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from sentence_transformers import SentenceTransformer, util

WORKSPACE_DICT = {
    'STORIES':f'{os.getcwd()}/stories',
    'SNAPSHOTS':f'{os.getcwd()}/snapshots',
    'PLOTS': f'{os.getcwd()}/plots',
    'EMBEDDINGS':f'{os.getcwd()}/embeddings'
}

SEP = '\t'

model = SentenceTransformer('all-mpnet-base-v2')

def create_workspace():
    for d in WORKSPACE_DICT.values():
        Path(d).mkdir(parents=True, exist_ok=True)

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

    if ('deleted' in id_dict.keys() and id_dict['deleted']) or ('dead' in id_dict.keys() and id_dict['dead']):
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

def save_story(story_id):
    """
    Save HackerNews story and all associated comments.
    """
    story_data = traverse_comment(id_dict=call_hackernews_api(f'item/{story_id}.json'), parent_='')
    write_ts = int(time.time())
    
    with open(f'{WORKSPACE_DICT["STORIES"]}/{story_id}_{write_ts}.csv', 'w') as f:
        f.write(story_data)    
    
def crawl():
    """
    Look thru the last 500 HackerNews stories and save a snapshot of stories and associated comments.
    """
    new_stories_list = call_hackernews_api('newstories.json')
    
    for s in new_stories_list:
        print(f'Saving {s}')
        save_story(s)

def save_file_to_s3(file_path, key, verbose=False):
    """
    Save file to S3 bucket
    """
    session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )

    s3 = session.resource('s3')
        
    try:
        if verbose:
            print(f'Saving {file_path}')
                
        s3.meta.client.upload_file(file_path, os.environ['AWS_HACKERNEWS_BUCKET'], key)
            
    except ClientError as e:
        print(e)
        
def save_directory_to_s3(directory, verbose=False):
    """
    Iterate thru a directory and save each file
    """
    session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )

    s3 = session.resource('s3')

    for f in os.listdir(directory):
        file_path = f'{directory}/{f}'        
        save_file_to_s3(file_path, f, verbose)


def create_crawl_snapshot(save_to_s3=False):
    """
    Compile all CSVs from base_directory and flatten them into one snapshot.
    """
    write_ts = int(time.time())
    raw_comments_key = f'hackernews_{write_ts}.csv'
    encoded_comments_key = f'hackernews_{write_ts}.pkl'
    
    raw_comments_filepath = f'{WORKSPACE_DICT["SNAPSHOTS"]}/{raw_comments_key}'
    encoded_comments_filepath = f'{WORKSPACE_DICT["EMBEDDINGS"]}/{encoded_comments_key}'

    df_list = []
    columns = ['timestamp', 'id', 'parent_id', 'type', 'by', 'text']
    
    for f in os.listdir(WORKSPACE_DICT["STORIES"]):
        if '.csv' in f:
            (story_id, write_ts) = f.replace('.csv', '').split('_')
            
            df = pd.read_csv(f'{WORKSPACE_DICT["STORIES"]}/{f}', names=columns, sep=SEP)
            df['story_id'] = story_id
            df['write_ts'] = write_ts
            df_list.append(df)

    merged_df = pd.concat(df_list).dropna(subset=['id'])
    merged_df = merged_df[['timestamp', 'story_id', 'id', 'parent_id', 'type', 'by', 'text', 'write_ts']].drop_duplicates()
    merged_df.to_csv(raw_comments_filepath, sep=SEP, index=False)

    merged_df['encoding'] = merged_df['text'].apply(lambda x: model.encode(x))
    merged_df.to_pickle(encoded_comments_filepath)

    if save_to_s3:
        save_file_to_s3(raw_comments_filepath, f'snapshots/{raw_comments_key}', verbose=True)
        save_file_to_s3(encoded_comments_filepath, f'embeddings/{encoded_comments_key}', verbose=True)
    
def crawl_to_s3():
    create_workspace()
    crawl()
    create_crawl_snapshot(save_to_s3=True)
    
if __name__ == '__main__':
    crawl_to_s3()
