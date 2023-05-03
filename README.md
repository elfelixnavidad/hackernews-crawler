Pull recent HackerNews comments & stories and save them to an S3 bucket.

Job Flow:
1. Pull list of all stories from hn endpoint: https://hacker-news.firebaseio.com/v0/newstories.json?print=pretty
2. Traverse comment tree for each comment within a story and save to a TSV with the story_id as the filename.
3. Flatten all comment TSVs into one within the snapshots folder.
4. Upload snapshot to S3.

First Run:
1. git clone https://github.com/elfelixnavidad/hackernews-crawler.git
2. cd hackernews-crawler
3. pip install virtualenv
4. virtualenv hn_crawler
5. source hn_crawler/bin/activate
6. pip install -r requirements.txt

.env:
List of env variables you need defined in order to run the project.