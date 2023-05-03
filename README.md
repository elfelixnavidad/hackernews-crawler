Project Overview & Job Flow
0. Pull recent HackerNews comments & stories and save them to an S3 bucket.
1. Pull list of all stories from hn endpoint: https://hacker-news.firebaseio.com/v0/newstories.json?print=pretty
2. Traverse comment tree for each comment within a story and save to a TSV with the story_id as the filename.
3. Flatten all comment TSVs into one within the snapshots folder.
4. Upload snapshot to S3.

First Run
git clone https://github.com/elfelixnavidad/hackernews-crawler.git
cd hackernews-crawler
pip install virtualenv
virtualenv hn_crawler
source hn_crawler/bin/activate
pip install -r requirements.txt

