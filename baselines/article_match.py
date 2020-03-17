import os
import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.misc_util import report_result


def get_article_bert(folder):
    article_bert = {}
    for line in open(os.path.join(folder, 'article_bert.tsv')):
        record = json.loads(line)
        article_bert[record['index']] = record['text']
    return article_bert


def get_tfidf_vectors(corpus):
    tfidf = TfidfVectorizer(ngram_range=(1, 1), lowercase=False)
    return tfidf.fit_transform(corpus)


def get_tfidf_dict(corpus):
    tfidf = TfidfVectorizer(ngram_range=(1, 1), lowercase=False)
    tfidf.fit(corpus)
    return tfidf


def get_article_tfidf_dict(article_dict):
    articles = []
    for aid, value in article_dict.items():
        bpe = value['a_bpe'] if value['a_bpe'] else value['t_bpe']
        articles.append(' '.join([str(b) for b in bpe]))
    return get_tfidf_dict(articles)


class ArticleMatch(object):
    def __init__(self, folder_path, ):
        self.folder = folder_path
        self.load_article_bert = False
        self.transformer = None
        self.authors = json.load(open(os.path.join(folder_path, 'frequent_author_record.json')))
        self.comments = json.load(open(os.path.join(folder_path, 'sentiment_comment.json')))
        self.articles = json.load(open(os.path.join(folder_path, 'article_idx.json')))
        self.y_name = ['vader', 'flair', 'blob_sentiment', 'blob_subjective']

    def build_transformer(self, load_article_bert):
        self.load_article_bert = load_article_bert
        if self.load_article_bert:
            self.transformer = get_article_bert(self.folder)
        else:
            self.transformer = get_article_tfidf_dict(self.articles)

    def get_vec(self, aids):
        vectors = []
        if self.load_article_bert:
            for aid in aids:
                vectors.append(self.transformer[aid.strip()])
            return vectors
        else:
            for aid in aids:
                value = self.articles[aid]
                bpe = value['a_bpe'] if value['a_bpe'] else value['t_bpe']
                vectors.append(' '.join([str(b) for b in bpe]))
            return self.transformer.transform(vectors)

    def article_match(self, pre_count, save_file=''):
        result = {k: [[], []] for k in self.y_name}
        result['author'] = []
        for au_id, track in self.authors.items():
            tmp_track = track[: -2]
            if 0 < pre_count < len(tmp_track):
                tmp_track = tmp_track[-pre_count:]
            tmp_aid = []
            for cid in tmp_track:
                tmp_aid.append(self.comments[cid]['aid'])
            history = self.get_vec(tmp_aid)
            tar_aid = [self.comments[track[-1]]['aid']]
            current = self.get_vec(tar_aid)
            max_idx = cosine_similarity(history, current).argmax()
            pre_cid = tmp_track[max_idx]
            for key in self.y_name:
                result[key][0].append(self.comments[pre_cid]['sentiment'][key])
                result[key][1].append(self.comments[track[-1]]['sentiment'][key])
            result['author'].append(au_id)
        report_result(result, self.y_name, save_file)


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for outlet in [
        'Archiveis',
        'DailyMail',
        'foxnews',
        'NewYorkTimes',
        'theguardian',
        'wsj'
    ]:
        folder = os.path.join('d:/data/outlets', outlet)
        print("Working on {} ...".format(outlet))
        am = ArticleMatch(folder_path=folder)
        print('TFIDF')
        am.build_transformer(load_article_bert=False)
        am.article_match(pre_count=12,
                         save_file=os.path.join(folder,
                                                'baseline_article_tfidf_12_result.txt'))
        print('BERT')
        am.build_transformer(load_article_bert=True)
        am.article_match(pre_count=12,
                         save_file=os.path.join(folder,
                                                'baseline_article_bert_12_result.txt'))
        del am
        print("Finish at {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        print()
