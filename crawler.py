#!/usr/bin/env python3
import json
import time

from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import pandas as pd
import numpy as np

BASE_DIRECTORY = './hackernews'

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
    if 'time' in id_dict.keys():
        post_timestamp = id_dict['time']

    post_id = ''
    if 'id' in id_dict.keys():
        post_id = id_dict['id']

    post_type = ''
    if 'type' in id_dict.keys():
        post_type = id_dict['type']

    post_by = ''
    if 'by' in id_dict.keys():
        post_by = id_dict['by']

    post_text = ''
    if post_type == 'comment':
        if 'text' in id_dict.keys():
            post_text = id_dict['text']
    elif post_type == 'story':
        if 'title' in id_dict.keys():
            post_text += id_dict['title']
        if 'url' in id_dict.keys():
            post_text += '|' + id_dict['url']
        if 'score' in id_dict.keys():
            post_text += '|' + str(id_dict['score'])
    post_text = "\"" + post_text + "\""
            
    return f'{post_timestamp}\t{post_id}\t{parent_id}\t{post_type}\t{post_by}\t{post_text}'

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
    with open(f'{BASE_DIRECTORY}/{story_id}/{story_id}_{write_ts}.tsv', 'w') as f:
        f.write(story_data)    
    
def main():
    new_stories_list = call_hackernews_api('newstories.json')

    Path(f'{BASE_DIRECTORY}').mkdir(parents=True, exist_ok=True)
    for s in new_stories_list:
        print(f'Saving {s}')
        save_story(s)
    
if __name__ == '__main__':
    main()
