import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.high_dev_examples, self.low_dev_examples = data_io.build_eval_examples(
            split='dev', min_cnt=config['min_comment_cnt'], pre_cnt=config['previous_comment_cnt'])
        self.high_dev_iter = data_io.build_iter_idx(self.high_dev_examples)
        self.low_dev_iter = data_io.build_iter_idx(self.low_dev_examples)
        self.test_examples, _ = data_io.build_eval_examples(
            split='test', min_cnt=-1, pre_cnt=config['previous_comment_cnt'])
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
        self.train_log = "Epoch {0} Prog {1:.1f}% || v_tar: {2:.3f} || f_tar: {3:.3f} || s_tar: {4:.3f} " \
                         "|| b_tar: {5:.3f} || e_tar: {6:.3f} || Author: {7:.3f} || t_track: {8:.3f} || " \
                         "v_track: {9:.3f} || f_track: {10:.3f} || s_track: {11:.3f} || b_track: {12:.3f} || " \
                         "e_track: {13:.3f} . "
        self.train_loss = {'author': [], 'v_pf': [], 'f_pf': [], 's_pf': [], 'b_pf': [], 'e_pf': [],
                           't_ax': [], 'v_ax': [], 'f_ax': [], 's_ax': [], 'b_ax': [], 'e_ax': []}

        self.dev_perf = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,
                         'emotion': 0, 'mean': 0}
        self.test_perf = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,
                          'emotion': 0, 'mean': 0}
        self.train_procedure = []

    def get_fp_loss(self, fp_result, fp_batch, loss_dict):
        loss = 0
        if self.config['build_author_predict']:
            author_loss = F.cross_entropy(fp_result['author'],
                                          torch.tensor(fp_batch['author'], device=self.device))
            loss += 0.1 * author_loss
            loss_dict['author'].append(author_loss.item())

        if self.config['sentiment_fingerprinting']:
            vader_loss = F.cross_entropy(fp_result['vader'],
                                         torch.tensor(fp_batch['v_tar'], device=self.device))
            loss += vader_loss
            loss_dict['v_pf'].append(vader_loss.item())

            flair_loss = F.cross_entropy(fp_result['flair'],
                                         torch.tensor(fp_batch['f_tar'], device=self.device))
            loss += flair_loss
            loss_dict['f_pf'].append(flair_loss.item())

            sent_loss = F.cross_entropy(fp_result['sent'],
                                        torch.tensor(fp_batch['s_tar'], device=self.device))
            loss += sent_loss
            loss_dict['s_pf'].append(sent_loss.item())

            subj_loss = F.cross_entropy(fp_result['subj'],
                                        torch.tensor(fp_batch['b_tar'], device=self.device))
            loss += subj_loss
            loss_dict['b_pf'].append(subj_loss.item())

        if self.config['emotion_fingerprinting']:
            emotion_loss = F.binary_cross_entropy_with_logits(fp_result['emotion'],
                                                              torch.tensor(fp_batch['e_tar'],
                                                                           device=self.device,
                                                                           dtype=torch.float))
            loss += emotion_loss
            loss_dict['e_pf'].append(emotion_loss.item())
        return loss

    def get_aux_loss(self, batch_idx, examples, loss_dict):
        loss = 0
        a_batch = self.data_io.get_batch_auxiliary(batch_idx, examples)

        if self.config['build_topic_predict']:
            result = self.model(is_auxiliary=True,
                                x=a_batch['article'],
                                target=['topic'], device=self.device)
            topic_loss = F.cross_entropy(result['topic'],
                                         torch.tensor(a_batch['t_tar'], device=self.device))
            loss += topic_loss
            loss_dict['t_ax'].append(topic_loss.item())

        aux_targets = []
        if self.config['build_sentiment_predict']:
            aux_targets.extend(['vader', 'flair', 'sent', 'subj'])
        if self.config['build_emotion_predict']:
            aux_targets.append('emotion')
        if len(aux_targets) > 0:
            result = self.model(is_auxiliary=True,
                                x=a_batch['comment'],
                                target=aux_targets, device=self.device)
            if self.config['build_sentiment_predict']:
                a_v_loss = F.cross_entropy(result['vader'],
                                           torch.tensor(a_batch['v_tar'], device=self.device))
                loss += a_v_loss
                loss_dict['v_ax'].append(a_v_loss.item())

                f_v_loss = F.cross_entropy(result['flair'],
                                           torch.tensor(a_batch['f_tar'], device=self.device))
                loss += f_v_loss
                loss_dict['f_ax'].append(f_v_loss.item())

                s_v_loss = F.cross_entropy(result['sent'],
                                           torch.tensor(a_batch['s_tar'], device=self.device))
                loss += s_v_loss
                loss_dict['s_ax'].append(s_v_loss.item())

                b_v_loss = F.cross_entropy(result['subj'],
                                           torch.tensor(a_batch['b_tar'], device=self.device))
                loss += b_v_loss
                loss_dict['b_ax'].append(b_v_loss.item())

            if self.config['build_emotion_predict']:
                e_v_loss = F.binary_cross_entropy_with_logits(result['emotion'],
                                                              torch.tensor(a_batch['e_tar'],
                                                                           device=self.device,
                                                                           dtype=torch.float))
                loss += e_v_loss
                loss_dict['e_ax'].append(e_v_loss.item())
        return loss

    def get_result(self, batch_idx, examples):
        fp_batch = self.data_io.get_batch_fingerprint(batch_idx, examples)
        fp_result = self.model(is_auxiliary=False,
                               author=fp_batch['author'],
                               read_target=fp_batch['r_target'],
                               read_track=fp_batch['r_track'],
                               write_track=fp_batch['w_track'],
                               emotion_track=fp_batch['e_tra'],
                               sentiment_track=(fp_batch['v_tra'],
                                                fp_batch['f_tra'],
                                                fp_batch['s_tra'],
                                                fp_batch['b_tra']),
                               device=self.device)
        return fp_batch, fp_result

    def run(self):
        low_dev_perf = 0
        for e in range(self.epoch):
            grad_dict = {'layer': [], 'ave': [], 'max': []}
            train_examples = self.data_io.build_train_examples(pre_cnt=self.config['previous_comment_cnt'],
                                                               min_cnt=self.config['min_comment_cnt'])
            train_iter = self.data_io.build_iter_idx(train_examples, True)
            for i, batch_idx in enumerate(train_iter):
                if self.config['free_fp'] < self.global_step:
                    fp_batch, fp_result = self.get_result(batch_idx, train_examples)
                    fp_loss = self.get_fp_loss(fp_result, fp_batch, self.train_loss)
                else:
                    fp_loss = 0

                if self.config['build_auxiliary_task'] and self.config['freeze_aux'] > e:
                    aux_loss = self.get_aux_loss(batch_idx, train_examples, self.train_loss)
                else:
                    aux_loss = 0

                loss = fp_loss + aux_loss
                loss.backward()
                if self.config['track_grad']:
                    track_grad(self.model.named_parameters(),
                               grad_dict['layer'], grad_dict['ave'], grad_dict['max'])
                if self.global_step % self.config['update_iter'] == 0:
                    clip_grad_norm_(self.params, self.config['grad_clip'])
                    self.sgd.step()
                    self.sgd.zero_grad()
                self.global_step += 1

                if self.global_step % (self.config['check_step']
                                       * self.config['update_iter']) == 0:
                    train_log = self.train_log.format(
                        e, 100.0 * i / len(train_iter),
                        np.mean(self.train_loss['v_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['f_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['s_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['b_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['e_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['author'][-self.config['check_step']:]),
                        np.mean(self.train_loss['t_ax'][-self.config['check_step']:]),
                        np.mean(self.train_loss['v_ax'][-self.config['check_step']:]),
                        np.mean(self.train_loss['f_ax'][-self.config['check_step']:]),
                        np.mean(self.train_loss['s_ax'][-self.config['check_step']:]),
                        np.mean(self.train_loss['b_ax'][-self.config['check_step']:]),
                        np.mean(self.train_loss['e_ax'][-self.config['check_step']:]))
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
            # if self.config['check_grad']:
                # plt.bar(np.arange(len(grad_dict['max'])), grad_dict['max'], alpha=0.1, lw=1, color="c")
                # plt.bar(np.arange(len(grad_dict['max'])), grad_dict['ave'], alpha=0.1, lw=1, color="b")
                # plt.hlines(0, 0, len(grad_dict['ave']) + 1, lw=2, color="k")
                # plt.xticks(range(0, len(grad_dict['ave']), 1), grad_dict['layer'], rotation="vertical")
                # plt.xlim(left=0, right=len(grad_dict['ave']))
                # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
                # plt.xlabel("Layers")
                # plt.ylabel("average gradient")
                # plt.title("Gradient flow")
                # plt.grid(True)
                # plt.legend([Line2D([0], [0], color="c", lw=4),
                #             Line2D([0], [0], color="b", lw=4),
                #             Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
                # saved_path = os.path.join(self.config['root_folder'], self.config['outlet'], 'grad_stat.png')
                # plt.savefig(saved_path)
                # plt.clf()

        json.dump(self.train_loss, open(os.path.join(
            self.config['root_folder'], self.config['outlet'], 'train_loss.json'), 'w'))
        with open(os.path.join(self.config['root_folder'],
                               self.config['outlet'], 'train_log.txt'), 'w') as f:
            for line in self.train_procedure:
                f.write(line + '\n')

    def get_perf(self, data_iter, examples):
        pred_records = []
        tmp_cnt = {'vader': [0, 0], 'flair': [0, 0], 'sent': [0, 0], 'subj': [0, 0],
                   'emotion': [0, 0]}
        acc = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,
               'emotion': 0, 'mean': 0}
        self.model.eval()
        for i, batch_idx in enumerate(data_iter):
            fp_batch, fp_result = self.get_result(batch_idx, examples)

            for j, a in enumerate(fp_batch['author']):
                pred_record = {'vader': [0, 0], 'flair': [0, 0], 'sent': [0, 0], 'subj': [0, 0],
                               'emotion': [0, 0]}
                if self.config['sentiment_fingerprinting']:
                    pred_record['vader'][0] = fp_result['vader'][j].argmax().item()
                    pred_record['vader'][1] = fp_batch['v_tar'][j]
                    tmp_cnt['vader'][int(pred_record['vader'][0] == pred_record['vader'][1])] += 1

                    pred_record['flair'][0] = fp_result['flair'][j].argmax().item()
                    pred_record['flair'][1] = fp_batch['f_tar'][j]
                    tmp_cnt['flair'][int(pred_record['flair'][0] == pred_record['flair'][1])] += 1

                    pred_record['sent'][0] = fp_result['sent'][j].argmax().item()
                    pred_record['sent'][1] = fp_batch['s_tar'][j]
                    tmp_cnt['sent'][int(pred_record['sent'][0] == pred_record['sent'][1])] += 1

                    pred_record['subj'][0] = fp_result['subj'][j].argmax().item()
                    pred_record['subj'][1] = fp_batch['b_tar'][j]
                    tmp_cnt['subj'][int(pred_record['subj'][0] == pred_record['subj'][1])] += 1

                if self.config['emotion_fingerprinting']:
                    pred_emo = [int(i > 0) for i in fp_result['emotion'][j].detach().tolist()]
                    pred_record['emotion'][0] = pred_emo
                    pred_record['emotion'][1] = fp_batch['e_tar'][j]
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


def track_grad(named_parameters, layers, ave_grads, max_grads):
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
    # plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    # plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)


if __name__ == '__main__':
    print('test')
    print(np.mean([]))
