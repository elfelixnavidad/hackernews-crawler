FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

CMD mkdir /app/stories
CMD mkdir /app/snapshots
CMD mkdir /app/plots
CMD mkdir /app/embeddings
CMD python crawler.py