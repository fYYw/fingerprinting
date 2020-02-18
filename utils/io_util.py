import os
import json
import random
import collections
import numpy as np

import torch
from bpemb import BPEmb


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def build_vocab(documents, word2idx, idx2word, vocab_size, min_freq):
    counter = collections.Counter()
    for doc in documents:
        counter.update(doc.split())
    word_idx = len(word2idx)
    for key, count in counter.most_common(vocab_size):
        if count < min_freq:
            break
        word2idx[key] = word_idx
        idx2word[word_idx] = key
        word_idx += 1


class Vocab(object):
    def __init__(self,
                 vocab_path=None,
                 examples=None,
                 vocab_size=1e5,
                 min_freq=3,
                 unk='[UNK]',
                 pad='[PAD]'):
        self.unk_token = unk
        self.pad_token = pad
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        self.vocab_size = vocab_size
        self.min_freq = min_freq

        if os.path.isfile(vocab_path):
            self.word2idx = load_vocab(vocab_path)
            self.idx2word = {value: key for key, value in self.word2idx.items()}
            self.vocab_size = len(self.word2idx)
        else:
            print('Building vocabulary from training examples ...')
            if not examples:
                raise ValueError('Got empty training examples.')
            build_vocab(examples, self.word2idx, self.idx2word, self.vocab_size, self.min_freq)
            with open(vocab_path, 'w') as f:
                for word, idx in self.word2idx.items():
                    f.write(word)
                    f.write('\n')

    def add_special_token(self, token):
        idx = len(self.word2idx)
        self.word2idx[token] = idx
        self.idx2word[idx] = token


def pad_and_mask(single_example, max_len, pad_idx):
    if len(single_example) < max_len:
        example = single_example + [single_example[-1] if pad_idx < 0 else pad_idx
                                    for _ in range(max_len - len(single_example))]
        mask = [1 for _ in range(len(single_example))] + [0 for _ in range(max_len - len(single_example))]
    else:
        example = single_example[:max_len]
        mask = [1 for _ in range(len(example))]
    return example, mask


def pad_seq(no_pad_examples, pad_idx, max_word_seq=-1):
    if max_word_seq < 0:
        max_word_seq = float('inf')
    max_word_len = max(len(e) for e in no_pad_examples)
    max_word_len = min(max_word_len, max_word_seq)
    padded_words, full_mask = [], []
    for words in no_pad_examples:
        padded_word, words_mask = pad_and_mask(words, max_word_len, pad_idx)
        padded_words.append(padded_word)
        full_mask.append(words_mask)
    return padded_words, full_mask


def pad_seq_seq(no_pad_examples, pad_idx, max_word_seq=-1, max_track_seq=-1):
    if max_word_seq < 0:
        max_word_seq = float('inf')
    if max_track_seq < 0:
        max_track_seq = float('inf')

    max_track_len, max_word_len = 0, 0
    for track in no_pad_examples:
        max_track_len = max(max_track_len, len(track))
        for words in track:
            max_word_len = max(max_track_len, len(words))
    max_word_len = min(max_word_len, max_word_seq)
    max_track_len = min(max_track_len, max_track_seq)

    padded_examples, full_mask, track_mask = [], [], []
    for track in no_pad_examples:
        words_masks, padded_words = [], []
        for words in track:
            padded_word, words_mask = pad_and_mask(words, max_word_len, pad_idx)
            words_masks.append(words_mask)
            padded_words.append(padded_word)
        if len(padded_words) < max_track_len:
            track_mask.append([1 for _ in range(len(padded_words))] +
                              [0 for _ in range(max_track_len - len(padded_words))])
            padded_words = padded_words + [padded_words[-1]
                                           for _ in range(max_track_len -
                                                          len(padded_words))]
            words_masks = words_masks + [words_masks[-1]
                                         for _ in range(max_track_len -
                                                        len(words_masks))]
        else:
            padded_words = padded_words[:max_track_len]
            words_masks = words_masks[:max_track_len]
            track_mask.append([1 for _ in range(len(padded_words))])
        padded_examples.append(padded_words)
        full_mask.append(words_masks)
    return padded_examples, full_mask, track_mask


def target_vectorization(sentiments):
    vader, flair = [], []
    blob_sent, blob_subj = [], []
    for s in sentiments:
        vader.append(s['vader'])
        flair.append(s['flair'])
        blob_sent.append(s['blob_sentiment'])
        blob_subj.append(s['blob_subjective'])
    return vader, flair, blob_sent, blob_subj


class IO(object):
    def __init__(self,
                 folder_path,
                 batch_size=64,
                 max_seq_len=128,
                 previous_comment_cnt=6,
                 min_comment_cnt=6,
                 target_sentiment=True,
                 target_emotion=False):
        self.authors = json.load(open(os.path.join(folder_path, 'frequent_author_record.json')))
        self.articles = json.load(open(os.path.join(folder_path, 'article_idx.json')))
        self.comments = json.load(open(os.path.join(folder_path, 'sentiment_emotion_comment.json')))
        self.topic_size = len(open(os.path.join(folder_path, 'vocab.topic')).readlines())
        self.word2idx = {}
        self.load_vocab(folder_path)

        self.batch_size = batch_size
        self.target_sentiment = target_sentiment
        self.target_emotion = target_emotion

        self.previous_cnt = previous_comment_cnt
        self.min_comment_cnt = min_comment_cnt
        self.max_seq_len = max_seq_len

    def load_vocab(self, folder):
        for line in open(os.path.join(folder, 'vocab.token'), encoding='utf-8'):
            key, idx = line.strip().split('\t')
            self.word2idx[key] = int(idx)

    def build_iter_idx(self, examples, shuffle=False):
        example_ids = list(examples.keys())
        batch_idx = np.arange(0, len(example_ids), self.batch_size)
        if shuffle:
            random.shuffle(example_ids)
        return [example_ids[i: i + self.batch_size] for i in batch_idx]

    def build_train_examples(self, pre_cnt, min_cnt=-1):
        examples, example_id = {}, 0
        for author, track in self.authors.items():
            tmp_track = track[: -2]
            if len(tmp_track) < min_cnt:
                continue
            for i in range(1, len(tmp_track)):
                examples[example_id] = (author, tmp_track[i], tmp_track[max(-pre_cnt + i, 0): i])
                example_id += 1
        return examples

    def build_eval_examples(self, pre_cnt, min_cnt, split='dev'):
        high_examples, low_examples = {}, {}
        for author, track in self.authors.items():
            if len(track) > min_cnt:
                target_examples = high_examples
            else:
                target_examples = low_examples
            cur_idx = len(target_examples)
            if split == 'dev':
                target_examples[cur_idx] = (author, track[-2], track[-pre_cnt - 2: -2])
            elif split == 'test':
                target_examples[cur_idx] = (author, track[-1], track[-pre_cnt - 1: -1])
            else:
                raise NotImplementedError()
        return high_examples, low_examples

    def get_batch_auxiliary(self, batch_idx, examples):
        batched_input = {}
        a_uni, c_uni = set(), set()
        article, comment = [], []
        t_tar, v_tar, f_tar, s_tar, b_tar, e_tar = [], [], [], [], [], []
        for idx in batch_idx:
            for cid in examples[idx][2]:
                if cid not in c_uni:
                    c_uni.add(cid)
                    comment.append(self.comments[cid]['bpe'])
                    v_tar.append(self.comments[cid]['sentiment']['vader'])
                    f_tar.append(self.comments[cid]['sentiment']['flair'])
                    s_tar.append(self.comments[cid]['sentiment']['blob_sentiment'])
                    b_tar.append(self.comments[cid]['sentiment']['blob_subjective'])
                    e_tar.append(self.comments[cid]['emotion'])

                    aid = self.comments[cid]['aid']
                    if aid not in a_uni:
                        article.append(self.articles[aid]['bpe'])
                        t_tar.append(self.articles[aid]['topic'])
        batched_input['article'] = pad_seq(article, pad_idx=0, max_word_seq=self.max_seq_len)
        batched_input['t_tar'] = t_tar
        batched_input['comment'] = pad_seq(comment, pad_idx=0, max_word_seq=self.max_seq_len)
        batched_input['v_tar'] = v_tar
        batched_input['f_tar'] = f_tar
        batched_input['s_tar'] = s_tar
        batched_input['b_tar'] = b_tar
        batched_input['e_tar'] = e_tar
        return batched_input

    def get_batch_fingerprint(self, batch_idx, examples):
        batched_input = {}
        author, r_tracks, w_tracks, r_target, w_target = [], [], [], [], []
        v_tar, f_tar, s_tar, b_tar, e_tar = [], [], [], [], []
        v_tra, f_tra, s_tra, b_tra, e_tra = [], [], [], [], []
        for idx in batch_idx:
            example = examples[idx]
            cid_tar = example[1]
            pid = self.comments[cid_tar]['pid']
            aid = self.comments[cid_tar]['aid']
            if pid and pid in self.comments:
                r_target.append(self.comments[pid]['bpe'])
            elif pid == 'N' or not pid:
                r_target.append(self.articles[aid]['bpe'])
            else:
                continue
            author.append(int(example[0]))
            w_target.append(self.comments[cid_tar]['bpe'])
            v_tar.append(self.comments[cid_tar]['sentiment']['vader'])
            f_tar.append(self.comments[cid_tar]['sentiment']['flair'])
            s_tar.append(self.comments[cid_tar]['sentiment']['blob_sentiment'])
            b_tar.append(self.comments[cid_tar]['sentiment']['blob_subjective'])
            e_tar.append(self.comments[cid_tar]['emotion'])

            r_track, w_track = [], []
            vader, flair, sent, subj, emotion = [], [], [], [], []
            for cid in example[2]:
                pid = self.comments[cid]['pid']
                aid = self.comments[cid]['aid']
                w_track.append(self.comments[cid]['bpe'])
                if pid and pid in self.comments:
                    r_track.append(self.comments[pid]['bpe'])
                elif pid == 'N' or not pid:
                    r_track.append(self.articles[aid]['bpe'])
                elif pid and pid not in self.comments:
                    r_track.append([2])
                else:
                    print('Unseen PID', pid)
                    raise NotImplementedError()
                vader.append(self.comments[cid]['sentiment']['vader'])
                flair.append(self.comments[cid]['sentiment']['flair'])
                sent.append(self.comments[cid]['sentiment']['blob_sentiment'])
                subj.append(self.comments[cid]['sentiment']['blob_subjective'])
                emotion.append(self.comments[cid]['emotion'])
            r_tracks.append(r_track)
            w_tracks.append(w_track)
            v_tra.append(vader)
            f_tra.append(flair)
            s_tra.append(sent)
            b_tra.append(subj)
            e_tra.append(emotion)
        batched_input['author'] = author
        batched_input['r_track'] = pad_seq_seq(r_tracks, pad_idx=0,
                                               max_word_seq=self.max_seq_len, max_track_seq=-1)
        batched_input['w_track'] = pad_seq_seq(w_tracks, pad_idx=0,
                                               max_word_seq=self.max_seq_len, max_track_seq=-1)
        batched_input['v_tra'] = pad_seq(v_tra, -1, max_word_seq=-1)[0]
        batched_input['f_tra'] = pad_seq(f_tra, -1, max_word_seq=-1)[0]
        batched_input['s_tra'] = pad_seq(s_tra, -1, max_word_seq=-1)[0]
        batched_input['b_tra'] = pad_seq(b_tra, -1, max_word_seq=-1)[0]
        batched_input['e_tra'] = pad_seq(e_tra, -1, max_word_seq=-1)[0]
        batched_input['r_target'] = pad_seq(r_target, pad_idx=0, max_word_seq=self.max_seq_len)
        batched_input['w_target'] = pad_seq(w_target, pad_idx=0, max_word_seq=self.max_seq_len)
        batched_input['v_tar'] = v_tar
        batched_input['f_tar'] = f_tar
        batched_input['s_tar'] = s_tar
        batched_input['b_tar'] = b_tar
        batched_input['e_tar'] = e_tar
        return batched_input


if __name__ == '__main__':
    io = IO(folder_path='e:/outlets/Archiveis')
    # examples = io.build_examples(1., 'train')
    # iter_idx = io.build_iter_idx(examples, True)
    # for batch_idx in iter_idx:
    #     io.fingerprinting_input(batch_idx, examples)
    #     io.topic_classification_input(batch_idx, examples)
    # bpe = BPEmb(lang='en', vs=50000)
    # tokens = bpe.encode('If you are using word embeddings like word2vec or GloVe, '
    #                     'you have probably encountered out-of-vocabulary words')
    # print(tokens)
    # words = bpe.decode(tokens)
    # print(words)
    t = [1, 2, 3, 4]
    print(t[1:-2])
