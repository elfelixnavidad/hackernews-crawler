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

import MySQLdb
from sqlalchemy import create_engine, types

BASE_DIRECTORY = './hackernews'
SNAPSHOT_DIRECTORY = './snapshots'
SEP = '\t'

def call_hackernews_api(endpoint):
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

    # post_text = re.sub(r'[^a-zA-Z0-9 .!?,]', '', post_text)
    # post_text = html.unescape(post_text)
    # post_text = "\"" + post_text + "\""
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
    data = create_row(id_dict, parent_)
    
    if 'kids' not in id_dict.keys():
        return data + '\n'
    else:
        for k in id_dict['kids']:
            data += '\n' + traverse_comment(call_hackernews_api(f'item/{k}.json'), id_dict['id'])

    return data

def save_story(story_id):
    story_data = traverse_comment(id_dict=call_hackernews_api(f'item/{story_id}.json'), parent_='')
    write_ts = int(time.time())
    
    Path(f'{BASE_DIRECTORY}/{story_id}').mkdir(parents=True, exist_ok=True)
    with open(f'{BASE_DIRECTORY}/{story_id}/{story_id}_{write_ts}.csv', 'w') as f:
        f.write(story_data)    
    
def crawl():
    Path(f'{BASE_DIRECTORY}').mkdir(parents=True, exist_ok=True)
    
    new_stories_list = call_hackernews_api('newstories.json')
    
    for s in new_stories_list:
        print(f'Saving {s}')
        save_story(s)

def create_crawl_snapshot():
    Path(f'{SNAPSHOT_DIRECTORY}').mkdir(parents=True, exist_ok=True)

    write_ts = int(time.time())
    snapshot_file = f'{SNAPSHOT_DIRECTORY}/hackernews_{write_ts}.csv'

    df_list = []
    columns = ['timestamp', 'id', 'parent_id', 'type', 'by', 'text']
    
    for f1 in os.listdir(BASE_DIRECTORY):
        for f2 in os.listdir(f'{BASE_DIRECTORY}/{f1}'):
            df = pd.read_csv(f'{BASE_DIRECTORY}/{f1}/{f2}', names=columns, sep=SEP)
            df['story_id'] = df.query('type == \'story\'')['id'].iloc[0]
            
            df_list.append(df)

    merged_df = pd.concat(df_list)\
                  .drop_duplicates()\
                  .dropna(subset=['id'])
    merged_df = merged_df[['timestamp', 'story_id', 'id', 'parent_id', 'type', 'by', 'text']]
    merged_df.to_csv(snapshot_file, sep=SEP, index=False)

def write_csv_to_table(csv_name, table_name):
    host = os.environ['PLANETSCALE_HOST']
    username = os.environ['PLANETSCALE_USERNAME']
    password = os.environ['PLANETSCALE_PASSWORD']
    db = os.environ['PLANETSCALE_DATABASE']

    connection_string = f'mysql+mysqldb://{username}:{password}@{host}/{db}'
    engine = create_engine(connection_string, echo=True)

    pd.read_csv(csv_name, sep=SEP)\
      .to_sql(table_name, con=engine, index=False, if_exists='replace')

if __name__ == '__main__':        
    crawl()
    create_crawl_snapshot()    
