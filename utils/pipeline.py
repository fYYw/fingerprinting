import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_


class Pipeline(object):
    def __init__(self,
                 data_io,
                 epoch,
                 model,
                 sgd,
                 config,
                 device):
        self.data_io = data_io
        self.high_dev_examples, self.low_dev_examples = data_io.build_split_examples(
            'dev', min_count=config['previous_comment_cnt'])
        self.high_dev_iter = data_io.build_iter_idx(self.high_dev_examples)
        self.low_dev_iter = data_io.build_iter_idx(self.low_dev_examples)
        self.test_examples = data_io.build_examples(1., 'test')
        self.test_iter = data_io.build_iter_idx(self.test_examples)
        self.model = model
        self.params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params.append(param)
                print("Module {} with size {}".format(name, param.data.size()))
        self.sgd = sgd
        self.config = config
        self.device = device

        self.epoch = epoch
        self.global_step = 0
        self.train_log = "Epoch {0} Prog {1:.1f}% || Vader: {2:.3f} || Flair: {3:.3f} || Sent: {4:.3f} " \
                         "|| Subj: {5:.3f} || Emotion: {6:.3f} || Author: {7:.3f} || Topic: {8:.3f}. "
        self.train_loss = {'vader': [0], 'flair': [0], 'sent': [0], 'subj': [0],
                           'emotion': [0], 'author': [0], 'topic': [0]}
        self.dev_loss = {'vader': [0], 'flair': [0], 'sent': [0], 'subj': [0],
                         'emotion': [0], 'author': [0], 'topic': [0]}

        self.dev_perf = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,
                         'emotion': 0, 'mean': 0}
        self.test_perf = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,
                          'emotion': 0, 'mean': 0}
        self.train_procedure = []

    def run(self):
        low_dev_perf = 0
        for e in range(self.epoch):
            if self.config['use_entire_example_epoch'] > e:
                train_examples = self.data_io.build_entire_examples()
            else:
                train_examples = self.data_io.build_examples(1.0, split='train')
            train_iter = self.data_io.build_iter_idx(train_examples, True)
            for i, batch_idx in enumerate(train_iter):
                loss = self.get_loss(batch_idx, train_examples, self.train_loss)
                loss.backward()
                # plot_grad_flow(self.model.named_parameters())
                if self.global_step % self.config['update_iter'] == 0:
                    clip_grad_norm_(self.params, self.config['grad_clip'])
                    self.sgd.step()
                    self.sgd.zero_grad()
                self.global_step += 1
                self.get_loss(random.sample(self.high_dev_iter, 1)[0], self.high_dev_examples, self.dev_loss)

                if self.global_step % (self.config['check_step']
                                       * self.config['update_iter']) == 0:
                    train_log = self.train_log.format(
                        e, 100.0 * i / len(train_iter),
                        np.mean(self.train_loss['vader'][-self.config['check_step']:]),
                        np.mean(self.train_loss['flair'][-self.config['check_step']:]),
                        np.mean(self.train_loss['sent'][-self.config['check_step']:]),
                        np.mean(self.train_loss['subj'][-self.config['check_step']:]),
                        np.mean(self.train_loss['emotion'][-self.config['check_step']:]),
                        np.mean(self.train_loss['author'][-self.config['check_step']:]),
                        np.mean(self.train_loss['topic'][-self.config['check_step']:]))
                    print(train_log)
                    self.train_procedure.append(train_log)

                    high_dev_perf, _ = self.get_perf(self.high_dev_iter, self.high_dev_examples)
                    # print('HIGH DEV: ', high_dev_perf)

                    if high_dev_perf['mean'] > self.dev_perf['mean']:
                        self.dev_perf = high_dev_perf
                        low_dev_perf, _ = self.get_perf(self.low_dev_iter, self.low_dev_examples)

                        torch.save({'model': self.model.state_dict(),
                                    'adam': self.sgd.state_dict()},
                                   os.path.join(self.config['root_folder'],
                                                self.config['outlet'], 'best_model.pt'))
                        test_perf, test_preds = self.get_perf(self.test_iter, self.test_examples)
                        self.test_perf = test_perf

                        json.dump(test_perf, open(os.path.join(
                            self.config['root_folder'], self.config['outlet'], 'test_perf.json'), 'w'))
                        with open(os.path.join(self.config['root_folder'],
                                               self.config['outlet'], 'test_pred.jsonl'), 'w') as f:
                            for pred in test_preds:
                                data = json.dumps(pred)
                                f.write(data)

            print('BEST DEV: ', self.dev_perf)
            print('LOw DEV: ', low_dev_perf)
            print('BEST TEST: ', self.test_perf)
            # train_examples = self.data_io.build_examples(prob_to_full=self.config['prob_to_full'], split='train')
            # train_iter = self.data_io.build_iter_idx(train_examples, True)
        json.dump(self.train_loss, open(os.path.join(
            self.config['root_folder'], self.config['outlet'], 'train_loss.json'), 'w'))
        json.dump(self.dev_loss, open(os.path.join(
            self.config['root_folder'], self.config['outlet'], 'dev_loss.json'), 'w'))
        with open(os.path.join(self.config['root_folder'],
                               self.config['outlet'], 'train_log.txt'), 'w') as f:
            for line in self.train_procedure:
                f.write(line + '\n')

    def get_loss(self, batch_idx, examples, loss_dict):
        loss = 0
        if self.config['build_topic_predict']:
            articles, topics = self.data_io.topic_classification_input(batch_idx, examples)
        else:
            articles, topics = None, None

        author, r_tracks, w_tracks, sentiment, emotion = self.data_io.fingerprinting_input(
            batch_idx, examples)

        train_result = self.model(author, r_tracks, w_tracks, articles, sentiment, emotion, self.device)

        if self.config['build_author_predict']:
            author_loss = f.cross_entropy(train_result['author'],
                                          torch.tensor(author, device=self.device))
            loss += 0.1 * author_loss
            loss_dict['author'].append(author_loss.item())
        if self.config['build_topic_predict']:
            b, d = train_result['topic'].size()
            topic_loss = f.cross_entropy(train_result['topic'],
                                         torch.tensor(topics, device=self.device))
            loss += topic_loss
            loss_dict['topic'].append(topic_loss.item())
        if self.config['sentiment_fingerprinting']:
            vader_loss = f.cross_entropy(
                train_result['vader'],
                torch.tensor(sentiment[0][0], device=self.device)[:, -1])
            loss += vader_loss
            loss_dict['vader'].append(vader_loss.item())

            flair_loss = f.cross_entropy(
                train_result['flair'],
                torch.tensor(sentiment[1][0], device=self.device)[:, -1])
            loss += flair_loss
            loss_dict['flair'].append(flair_loss.item())

            blob_sent = f.cross_entropy(
                train_result['sent'],
                torch.tensor(sentiment[2][0], device=self.device)[:, -1])
            loss += blob_sent
            loss_dict['sent'].append(blob_sent.item())

            blob_subj = f.cross_entropy(
                train_result['subj'],
                torch.tensor(sentiment[3][0], device=self.device)[:, -1])
            loss += blob_subj
            loss_dict['subj'].append(blob_subj.item())
        if self.config['emotion_fingerprinting']:
            emotion_loss = f.binary_cross_entropy_with_logits(train_result['emotion'],
                                                              torch.tensor(emotion[0], device=self.device,
                                                                           dtype=torch.float)[:, -1])
            loss += emotion_loss
            loss_dict['emotion'].append(emotion_loss.item())
        return loss

    def get_perf(self, data_iter, examples):
        pred_records = []
        tmp_cnt = {'vader': [0, 0], 'flair': [0, 0], 'sent': [0, 0], 'subj': [0, 0],
                   'emotion': [0, 0]}
        acc = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,
               'emotion': 0, 'mean': 0}
        self.model.eval()
        for i, batch_idx in enumerate(data_iter):
            author, r_tracks, w_tracks, sentiment, emotion = self.data_io.fingerprinting_input(
                batch_idx, examples)
            articles, topics = None, None
            result = self.model(author, r_tracks, w_tracks, articles, sentiment, emotion, self.device, train=False)
            for j, a in enumerate(author):
                pred_record = {'vader': [0, 0], 'flair': [0, 0], 'sent': [0, 0], 'subj': [0, 0],
                               'emotion': [0, 0]}
                if self.config['sentiment_fingerprinting']:
                    pred_record['vader'][0] = result['vader'][j].argmax().item()
                    pred_record['vader'][1] = sentiment[0][0][j][-1]
                    tmp_cnt['vader'][int(pred_record['vader'][0] == pred_record['vader'][1])] += 1
                    pred_record['flair'][0] = result['flair'][j].argmax().item()
                    pred_record['flair'][1] = sentiment[1][0][j][-1]
                    tmp_cnt['flair'][int(pred_record['flair'][0] == pred_record['flair'][1])] += 1
                    pred_record['sent'][0] = result['sent'][j].argmax().item()
                    pred_record['sent'][1] = sentiment[2][0][j][-1]
                    tmp_cnt['sent'][int(pred_record['sent'][0] == pred_record['sent'][1])] += 1
                    pred_record['subj'][0] = result['subj'][j].argmax().item()
                    pred_record['subj'][1] = sentiment[3][0][j][-1]
                    tmp_cnt['subj'][int(pred_record['subj'][0] == pred_record['subj'][1])] += 1
                if self.config['emotion_fingerprinting']:
                    pred_emo = [int(i > 0) for i in result['emotion'][j].detach().tolist()]
                    pred_record['emotion'][0] = pred_emo
                    pred_record['emotion'][1] = emotion[0][j][-1]
                    tmp_cnt['emotion'][0] += sum([pred == gold for pred, gold in zip(*pred_record['emotion'])])
                    tmp_cnt['emotion'][1] += sum([pred != gold for pred, gold in zip(*pred_record['emotion'])])
                pred_records.append(pred_record)
        self.model.train()
        tmp_acc_sum = []
        if self.config['sentiment_fingerprinting']:
            for key in ['vader', 'flair', 'sent', 'subj']:
                acc[key] = 1.0 * tmp_cnt[key][1] / (tmp_cnt[key][0] + tmp_cnt[key][1])
                tmp_acc_sum.append(acc[key])
        if self.config['emotion_fingerprinting']:
            acc['emotion'] = 1.0 * tmp_cnt['emotion'][0] / (tmp_cnt['emotion'][0] + tmp_cnt['emotion'][1])
            tmp_acc_sum.append(acc['emotion'])
        acc['mean'] = sum(tmp_acc_sum) / len(tmp_acc_sum)
        return acc, pred_records


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    max_grads = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


if __name__ == '__main__':
    print('test')
    print(np.mean([]))
