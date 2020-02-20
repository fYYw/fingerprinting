import os
import json
import time
from collections import Counter


class BaseLine(object):
    def __init__(self,
                 folder_path):
        self.authors = json.load(open(os.path.join(folder_path, 'frequent_author_record.json')))
        self.comments = json.load(open(os.path.join(folder_path, 'sentiment_comment.json')))
        self.articles = json.load(open(os.path.join(folder_path, 'article_idx.json')))

    def author_rating_frequency(self, pre_count):
        predict = {}
        for au_id, track in self.authors.items():
            pass


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for outlet in [
        'archiveis'
        # 'theguardian',
        # 'washingtonpost',
        # 'wsj',
        # 'NewYorkTimes'
    ]:  # os.listdir('e:/outlets'): #['Archiveis']:
        print("Working on {} ...".format(outlet))
        folder = os.path.join('e:/outlets', outlet)
