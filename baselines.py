import os
import json
import time
from collections import Counter
from utils.misc_util import report_result


class BaseLine(object):
    def __init__(self,
                 folder_path):
        self.authors = json.load(open(os.path.join(folder_path, 'frequent_author_record.json')))
        self.comments = json.load(open(os.path.join(folder_path, 'sentiment_comment.json')))
        self.articles = json.load(open(os.path.join(folder_path, 'article_idx.json')))
        self.y_name = ['vader', 'flair', 'blob_sentiment', 'blob_subjective']

    def author_rating_frequency(self, pre_count, save_file=''):
        predict = {}
        result = {k: [[], []] for k in self.y_name}
        result['author'] = []
        for au_id, track in self.authors.items():
            tmp_track = track[: -2]
            if 0 < pre_count < len(tmp_track):
                tmp_track = tmp_track[-pre_count:]
            predict[au_id] = {k: [] for k in self.y_name}
            for cid in tmp_track:
                comment = self.comments[cid]
                for y_name in self.y_name:
                    predict[au_id][y_name].append(comment['sentiment'][y_name])
            for key, value in predict[au_id].items():
                author_track_predict = Counter(value).most_common(1)[0][0]
                result[key][0].append(author_track_predict)
                result[key][1].append(self.comments[track[-1]]['sentiment'][key])
            result['author'].append(au_id)
        report_result(result, self.y_name, save_file)


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for outlet in os.listdir('d:/data/outlets'):  # [
        # 'archiveis'
        # 'theguardian',
        # 'wsj',
        # 'NewYorkTimes'
        # ]:
        print("Working on {} ...".format(outlet))
        folder = os.path.join('d:/data/outlets', outlet)
        base = BaseLine(folder)
        base.author_rating_frequency(12,
                                     os.path.join(folder, 'baseline_author_12_frequency_result.txt'))
