import os
import re
import json
import time
import numpy as np
from bpemb import BPEmb
from datetime import datetime
import unicodedata

ROOT = 'D:/data/outlets'
bpemb_en = BPEmb(lang='en', vs=25000)


def gen_author_record(comment_dict, percentile=70, min_count=10):
    author_record = {}
    for cid, value in comment_dict.items():
        author = value['author']
        if author not in author_record:
            author_record[author] = []
        if value['content']:
            author_record[author].append((cid, datetime.strptime(value['time'], '%Y-%m-%d %H:%M:%S')))

    author_comment_counts = []
    for author, comments in author_record.items():
        author_comment_counts.append(len(comments))

    filter_count = np.percentile(author_comment_counts, percentile)
    print('Setting minimum count', max(filter_count, min_count))
    author_idx_record, author2idx, author_idx = {}, {}, 0
    for author in list(author_record.keys()):
        comments = author_record[author]
        if len(comments) < max(filter_count, min_count):
            del author_record[author]
        else:
            if author not in author2idx:
                author2idx[author] = author_idx
                author_idx += 1
            sorted_comment = sorted(author_record[author], key=lambda a: a[1])
            author_idx_record[author2idx[author]] = [cid for (cid, _) in sorted_comment]
    return author_idx_record


def bpe_article_comments(folder):
    pure_article_dict, pure_comment_dict = {}, {}
    article_comment_file = os.path.join(folder, 'article_comment.json')
    article_comment_dict = json.load(open(article_comment_file))
    for aid, value in article_comment_dict.items():
        article_dict = value['article']
        pure_article_dict[aid] = {'time': article_dict['time'],
                                  'title': article_dict['title'],
                                  'bpe': ' '.join(bpemb_en.encode(unicodedata.normalize('NFD', article_dict['title']))),
                                  'topic': article_dict['topic'], 'outlet': article_dict['outlet'],
                                  'category': article_dict['category']}
        for cid, c_value in value['comment'].items():
            if c_value['time'] != 'N' and c_value['content']:
                pure_comment_dict[cid] = {'time': c_value['time'], 'article_id': aid,
                                          'author': unicodedata.normalize('NFD', c_value['author']),
                                          'bpe': ' '.join(bpemb_en.encode(
                                              unicodedata.normalize('NFD', c_value['content']))),
                                          'content': c_value['content'], 'pid': c_value['pid']}
    json.dump(pure_article_dict, open(os.path.join(folder, 'pure_article.json'), 'w'))
    json.dump(pure_comment_dict, open(os.path.join(folder, 'pure_comment.json'), 'w'))


def gen_examples(folder, percentile, min_count):
    article_file = os.path.join(folder, 'pure_article.json')  # key: time, title, topic, outlet, category, bpe
    article_dict = json.load(open(article_file))
    comment_file = os.path.join(folder, 'pure_comment.json')  # key: time, author, article_id, content, pid, bpe
    comment_dict = json.load(open(comment_file))
    bpe_word2idx = {}
    for line in open('d:/data/en.wiki.bpe.vs25000.vocab', encoding='utf-8'):
        key, idx = line.strip().split('\t')
        bpe_word2idx[key] = idx

    author_record = gen_author_record(comment_dict, percentile, min_count)
    comment_idx, article_idx = {}, {}
    word2idx, word_idx = {'[PAD]': 0, '[UNK]': 1, '[EMPTY]': 2}, 3
    topic2idx, topic_idx = {}, 0
    for aid, a_value in article_dict.items():
        if a_value['topic'] not in topic2idx:
            topic2idx[a_value['topic']] = topic_idx
            topic_idx += 1
        for token in a_value['bpe'].split():
            if token not in word2idx and token in bpe_word2idx:
                word2idx[token] = word_idx
                word_idx += 1
        article_idx[aid] = {'bpe': [word2idx[token] if token in word2idx else 1
                                    for token in a_value['bpe'].split()],
                            'topic': topic2idx[a_value['topic']]}

    for cid, c_value in comment_dict.items():
        for token in c_value['bpe'].split():
            if token not in word2idx and token in bpe_word2idx:
                word2idx[token] = word_idx
                word_idx += 1
        comment_idx[cid] = {'bpe': [word2idx[token] if token in word2idx else 1
                                    for token in c_value['bpe'].split()],
                            'pid': c_value['pid'],
                            'aid': c_value['article_id']}

    with open(os.path.join(folder, 'vocab.token'), 'w', encoding='utf-8') as f:
        for key, value in word2idx.items():
            f.write('{}\t{}\n'.format(key, value))
    with open(os.path.join(folder, 'vocab.topic'), 'w', encoding='utf-8') as f:
        for key, value in topic2idx.items():
            f.write('{}\t{}\n'.format(key, value))
    json.dump(author_record,
              open(os.path.join(folder, 'frequent_author_record.json'), 'w'))
    print('Unique authors: ', len(author_record))
    json.dump(article_idx,
              open(os.path.join(folder, 'article_idx.json'), 'w'))
    json.dump(comment_idx,
              open(os.path.join(folder, 'comment_idx.json'), 'w'))


def add_sentiment_score(folder):
    sentiment_file = os.path.join(folder, 'batch_sentiment.json')
    sentiment_dict = json.load(open(sentiment_file))
    example_file = os.path.join(folder, 'comment_idx.json')
    example_dict = json.load(open(example_file))
    for cid in list(example_dict.keys()):
        if cid in sentiment_dict:
            if sentiment_dict[cid]['vader']['compound'] > 0:
                vader = 1
            elif sentiment_dict[cid]['vader']['compound'] < 0:
                vader = 2
            else:
                vader = 0

            if sentiment_dict[cid]['blob'][0] > 0:
                blob_sentiment = 1
            elif sentiment_dict[cid]['blob'][0] < 0:
                blob_sentiment = 2
            else:
                blob_sentiment = 0

            if sentiment_dict[cid]['blob'][1] > 0.5:
                blob_subjective = 1
            elif sentiment_dict[cid]['blob'][1] < 0.5:
                blob_subjective = 2
            else:
                blob_subjective = 0

            if sentiment_dict[cid]['flair'][0][1] == 'POSITIVE':
                flair = 1
            elif sentiment_dict[cid]['flair'][0][1] == 'NEGATIVE':
                flair = 2
            else:
                flair = 0
            example_dict[cid]['sentiment'] = {'vader': vader, 'flair': flair,
                                              'blob_sentiment': blob_sentiment,
                                              'blob_subjective': blob_subjective}
        else:
            del example_dict[cid]
    json.dump(example_dict,
              open(os.path.join(folder, 'sentiment_comment.json'), 'w'))


def add_emotion_score(folder):
    emotion_file = os.path.join(folder, 'emotion.pred')
    emotion_scores = {}
    for i, line in enumerate(open(emotion_file)):
        if i == 0:
            title = line.strip().split('\t')[:-1:2]
            print(title)
            continue
        elements = line.strip().split('\t')[1::2][:-2]
        emotion_scores[i] = [int(float(e)) for e in elements]
    example_file = os.path.join(folder, 'sentiment_comment.json')
    example_dict = json.load(open(example_file))

    example_tsv = os.path.join(folder, 'comments.tsv')
    for i, line in enumerate(open(example_tsv, encoding='utf-8')):
        cid, text = line.strip().split('\t')
        if cid in example_dict:
            example_dict[cid]['emotion'] = emotion_scores[i]
        else:
            print(cid)
    json.dump(example_dict,
              open(os.path.join(folder, 'sentiment_emotion_comment.json'), 'w'))


def emotion_test(folder):
    # emotion_file = os.path.join(folder, 'emotion.pred')
    example_tsv = os.path.join(folder, 'comments.tsv')
    # print(len(open(emotion_file).readlines()))
    # print(len(open(example_tsv, encoding='utf').readlines()))
    for line in open(example_tsv, encoding='utf'):
        cid, text = line.strip().split('\t')
        t = re.sub(r'[^a-zA-Z0-9.?!]', '', text)
        if len(t) == 0:
            print(text)


def test_tsv():
    t = ['★', '“', '“', '§']


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for outlet in [
        # 'Archiveis',
        # 'cnn',
        # 'DailyMail',
        # 'foxnews',
        'NewYorkTimes',
        # 'theguardian',
        # 'washingtonpost',
        'wsj'
    ]:  # os.listdir(ROOT):
        print("Working on {} ...".format(outlet))
        # bpe_article_comments(os.path.join(ROOT, outlet))
        # gen_examples(os.path.join(ROOT, outlet), percentile=60, min_count=4)
        # add_sentiment_score(os.path.join(ROOT, outlet))
        # emotion_test(os.path.join(ROOT, outlet))
        add_emotion_score(os.path.join(ROOT, outlet))
        print("Finish at {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
