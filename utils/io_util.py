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


def full_records(author_dict, previous_cnt=6, split='train'):
    examples, example_idx = {}, 0
    for author, history in author_dict.items():
        if split == 'train':
            examples[example_idx] = (author, history[-previous_cnt - 2:-2])
        elif split == 'dev':
            examples[example_idx] = (author, history[-previous_cnt - 1:-1])
        else:
            examples[example_idx] = (author, history[-previous_cnt:])
        example_idx += 1
    return examples


def total_records(author_dict, previous_cnt=6):
    examples, example_idx = {}, 0
    for author, history in author_dict.items():
        tmp_history = history[: -2]
        for i in range(1, len(tmp_history)):
            examples[example_idx] = (author, tmp_history[max(-previous_cnt + i, 0): i + 1])
            example_idx += 1
    return examples


def full_records_frequency(author_dict, previous_cnt=6, min_cnt=6, split='dev'):
    high_examples, high_idx = {}, 0
    low_examples, low_idx = {}, 0

    for author, history in author_dict.items():
        if len(history) > min_cnt:
            if split == 'train':
                high_examples[high_idx] = (author, history[-previous_cnt - 2: -2])
            elif split == 'dev':
                high_examples[high_idx] = (author, history[-previous_cnt - 1: -1])
            else:
                high_examples[high_idx] = (author, history[-previous_cnt:])
            high_idx += 1
        else:
            if split == 'train':
                low_examples[low_idx] = (author, history[-previous_cnt - 2: -2])
            elif split == 'dev':
                low_examples[low_idx] = (author, history[-previous_cnt - 1: -1])
            else:
                low_examples[low_idx] = (author, history[-previous_cnt:])
            low_idx += 1
    return high_examples, low_examples


def sample_records(author_dict, previous_cnt=6, min_cnt=-1, split='train'):
    examples, example_idx = {}, 0
    if split == 'dev' or split == 'test':
        examples = full_records(author_dict, previous_cnt, split=split)
    else:
        author_cnt = [(author, len(cmts)) for author, cmts in author_dict.items() if len(cmts) > min_cnt]
        total_cnt = sum([a[1] for a in author_cnt])
        author_prob = [1.0 * cnt[1] / total_cnt for cnt in author_cnt]
        sampled_author = np.random.choice(author_cnt,
                                          size=len(author_cnt),
                                          p=author_prob)
        for (author, cnt) in sampled_author:
            history = author_dict[author][1:-2]  # dont account dev and test
            if len(history) < previous_cnt:
                examples[example_idx] = (author, history)
            else:
                random_idx = np.random.choice(range(1, len(history)), size=1)[0]
                examples[example_idx] = (author, history[max(0, random_idx - previous_cnt): random_idx + 1])
            example_idx += 1
    return examples


def pad_and_mask(single_example, max_len, pad_idx=-1):
    if len(single_example) < max_len:
        example = single_example + [pad_idx if pad_idx > -1 else single_example[-1]
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

    def build_examples(self, prob_to_full, split='train'):
        sampled_prob = np.random.choice(10, size=1)[0]
        if sampled_prob > prob_to_full * 10:
            print("Build samples by frequency ..")
            return sample_records(self.authors, self.previous_cnt, split)
        else:
            if self.min_comment_cnt > 1:
                print("Bulid high frequent examples ..")
                examples, _ = full_records_frequency(self.authors, previous_cnt=self.previous_cnt,
                                                     min_cnt=self.min_comment_cnt, split=split)
                return examples
            print("Bulid full examples ..")
            return full_records(self.authors, self.previous_cnt, split)

    def build_split_examples(self, split='dev', min_count=0):
        min_cnt = min_count if min_count > 1 else self.min_comment_cnt
        return full_records_frequency(self.authors, previous_cnt=self.previous_cnt,
                                      min_cnt=min_cnt, split=split)

    def build_entire_examples(self):
        return total_records(self.authors, previous_cnt=self.previous_cnt)

    def fingerprinting_input(self, batch_idx, examples):
        author, read_tracks, write_tracks, emotions = [], [], [], []
        vaders, flairs, sents, subjs = [], [], [], []
        for idx in batch_idx:
            example = examples[idx]
            author.append(int(example[0]))
            read_track, write_track = [], []
            vader, flair, sent, subj, emotion = [], [], [], [], []
            for cid in example[1]:
                pid = self.comments[cid]['pid']
                aid = self.comments[cid]['aid']
                if pid and pid in self.comments:
                    read_track.append(self.comments[pid]['bpe'])
                elif pid == 'N' or not pid:
                    read_track.append(self.articles[aid]['bpe'])
                elif pid and pid not in self.comments:
                    read_track.append([2])
                else:
                    print('Unseen PID', pid)
                    raise NotImplementedError()

                write_track.append(self.comments[cid]['bpe'])

                if self.target_sentiment:
                    vader.append(self.comments[cid]['sentiment']['vader'])
                    flair.append(self.comments[cid]['sentiment']['flair'])
                    sent.append(self.comments[cid]['sentiment']['blob_sentiment'])
                    subj.append(self.comments[cid]['sentiment']['blob_subjective'])
                if self.target_emotion:
                    emotion.append(self.comments[cid]['emotion'])
            read_tracks.append(read_track)
            write_tracks.append(write_track)
            vaders.append(vader)
            flairs.append(flair)
            sents.append(sent)
            subjs.append(subj)
            emotions.append(emotion)
        read_tracks = pad_seq_seq(read_tracks, pad_idx=0, max_word_seq=self.max_seq_len, max_track_seq=-1)
        write_tracks = pad_seq_seq(write_tracks, pad_idx=0, max_word_seq=self.max_seq_len, max_track_seq=-1)
        vaders = pad_seq(vaders, -1, max_word_seq=-1)
        flairs = pad_seq(flairs, -1, max_word_seq=-1)
        sents = pad_seq(sents, -1, max_word_seq=-1)
        subjs = pad_seq(subjs, -1, max_word_seq=-1)
        emotions = pad_seq(emotions, -1, max_word_seq=-1)
        return author, read_tracks, write_tracks, (vaders, flairs, sents, subjs), emotions

    def topic_classification_input(self, batch_idx, examples):
        article, topic = [], []
        unique_aid = set()
        for idx in batch_idx:
            for cid in examples[idx][1]:
                aid = self.comments[cid]['aid']
                if aid not in unique_aid:
                    unique_aid.add(aid)
                    topic.append(self.articles[aid]['topic'])
                    article.append(self.articles[aid]['bpe'])
        article = pad_seq(article, pad_idx=0, max_word_seq=self.max_seq_len)
        return article, topic


if __name__ == '__main__':
    io = IO(folder_path='e:/outlets/Archiveis')
    full_records_frequency(io.authors, previous_cnt=12, frequent_group=4)
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
