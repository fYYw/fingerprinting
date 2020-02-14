import os
import json
import time
from collections import Counter


def build_author_track(authors, comments, articles):
    author_score = {}
    for au_id, history in authors.items():
        a_score = {'a': {'vader': {'all': []}, 'flair': {'all': []},
                         'sent': {'all': []}, 'subj': {'all': []}},
                   'c': {'vader': {'all': []}, 'flair': {'all': []},
                         'sent': {'all': []}, 'subj': {'all': []}}}
        for cid in history[: -2]:
            pid = comments[cid]['pid']
            aid = comments[cid]['aid']
            topic = articles[aid]['topic']
            if topic not in a_score['a']['vader']:
                for key, value in a_score.items():
                    for model, track in value.items():
                        track[topic] = []
            if pid == 'N' or not pid:
                a_score['a']['vader'][topic].append(comments[cid]['sentiment']['vader'])
                a_score['a']['flair'][topic].append(comments[cid]['sentiment']['flair'])
                a_score['a']['sent'][topic].append(comments[cid]['sentiment']['blob_sentiment'])
                a_score['a']['subj'][topic].append(comments[cid]['sentiment']['blob_subjective'])
                a_score['a']['vader']['all'].append(comments[cid]['sentiment']['vader'])
                a_score['a']['flair']['all'].append(comments[cid]['sentiment']['flair'])
                a_score['a']['sent']['all'].append(comments[cid]['sentiment']['blob_sentiment'])
                a_score['a']['subj']['all'].append(comments[cid]['sentiment']['blob_subjective'])
            else:
                a_score['c']['vader'][topic].append(comments[cid]['sentiment']['vader'])
                a_score['c']['flair'][topic].append(comments[cid]['sentiment']['flair'])
                a_score['c']['sent'][topic].append(comments[cid]['sentiment']['blob_sentiment'])
                a_score['c']['subj'][topic].append(comments[cid]['sentiment']['blob_subjective'])
                a_score['c']['vader']['all'].append(comments[cid]['sentiment']['vader'])
                a_score['c']['flair']['all'].append(comments[cid]['sentiment']['flair'])
                a_score['c']['sent']['all'].append(comments[cid]['sentiment']['blob_sentiment'])
                a_score['c']['subj']['all'].append(comments[cid]['sentiment']['blob_subjective'])
        author_score[au_id] = a_score
    return author_score


def author_mean(author_scores, authors, comments, articles):
    all_pred = {'vader': [0, 0], 'flair': [0, 0], 'sent': [0, 0], 'subj': [0, 0]}
    top_pred = {'vader': [0, 0], 'flair': [0, 0], 'sent': [0, 0], 'subj': [0, 0]}
    for au_id, history in authors.items():
        if au_id not in author_scores:
             continue
        cid = history[-1]
        comment = comments[cid]
        pid = comment['pid']
        aid = comment['aid']
        topic = articles[aid]['topic']
        if pid == 'N' or not pid:
            if author_scores[au_id]['a']['vader']['all']:
                author_score_track = author_scores[au_id]['a']
            else:
                author_score_track = author_scores[au_id]['c']
        else:
            if author_scores[au_id]['c']['vader']['all']:
                author_score_track = author_scores[au_id]['c']
            else:
                author_score_track = author_scores[au_id]['a']

        for key1, key2 in zip(['vader', 'flair', 'sent', 'subj'],
                              ['vader', 'flair', 'blob_sentiment', 'blob_subjective']):
            score_all = Counter(author_score_track[key1]['all']).most_common(1)[0][0]
            if score_all == comment['sentiment'][key2]:
                all_pred[key1][1] += 1
            else:
                all_pred[key1][0] += 1

            if topic in author_score_track[key1] and author_score_track[key1][topic]:
                score_target = Counter(author_score_track[key1][topic]).most_common(1)[0][0]
            else:
                score_target = score_all
            if score_target == comment['sentiment'][key2]:
                top_pred[key1][1] += 1
            else:
                top_pred[key1][0] += 1
    all_res = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0, 'emotion': 0}
    top_res = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0, 'emotion': 0}
    all_mean = []
    for key, value in all_pred.items():
        all_res[key] = 1. * value[1] / (value[0] + value[1] + 1e-18)
        all_mean.append(all_res[key])
    all_res['mean'] = sum(all_mean) / len(all_mean)

    top_mean = []
    for key, value in top_pred.items():
        top_res[key] = 1. * value[1] / (value[0] + value[1] + 1e-18)
        top_mean.append(top_res[key])
    top_res['mean'] = sum(top_mean) / len(top_mean)
    print('ALL: ', all_res)
    print('TOPIC: ', top_res)


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for outlet in [
        # 'theguardian',
        # 'washingtonpost',
        'wsj',
        'NewYorkTimes'
                   ]: #os.listdir('e:/outlets'): #['Archiveis']:
        print("Working on {} ...".format(outlet))
        folder = os.path.join('e:/outlets', outlet)
        authors = json.load(open(os.path.join(folder, 'frequent_author_record.json')))
        comments = json.load(open(os.path.join(folder, 'sentiment_comment.json')))
        articles = json.load(open(os.path.join(folder, 'article_idx.json')))
        author_track = build_author_track(authors, comments, articles)
        author_mean(author_track, authors, comments, articles)
