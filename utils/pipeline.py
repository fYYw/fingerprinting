import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score

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
        self.train_log = "Epoch {0} Prog {1:.1f}% || Author: {2:.3f} || topic: {3:.3f} || v_tar: {4:.3f} " \
                         "|| v_track: {5:.3f} || f_tar: {6:.3f} || f_track: {7:.3f} || s_tar: {8:.3f} || " \
                         "s_track: {9:.3f} || b_tar: {10:.3f} || b_track: {11:.3f} || e_tar: {12:.3f} || " \
                         "e_track: {13:.3f} . "
        self.train_loss = {'author': [0], 'v_pf': [0], 'f_pf': [0], 's_pf': [0], 'b_pf': [0], 'e_pf': [0],
                           't_ax': [0], 'v_ax': [0], 'f_ax': [0], 's_ax': [0], 'b_ax': [0], 'e_ax': [0]}

        self.dev_perf = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,
                         'emotion': 0, 'mean': 0}
        self.test_perf = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,
                          'emotion': 0, 'mean': 0}
        self.train_procedure = []

        self.y_freq = {}

    def get_fp_loss(self, fp_result, fp_batch, loss_dict, un_freeze_fp, un_freeze_author):
        loss = 0
        if self.config['build_author_predict'] and un_freeze_author:
            author_loss = F.cross_entropy(fp_result['author'],
                                          torch.tensor(fp_batch['author'], device=self.device))
            loss += author_loss
            loss_dict['author'].append(author_loss.item())

        if self.config['vader'] and un_freeze_fp:
            v_tar = torch.tensor(fp_batch['v_tar'], device=self.device)
            v_even_idx = get_even_class(v_tar, device=self.device)
            if v_even_idx is not None:
                vader_loss = F.cross_entropy(fp_result['vader'].index_select(0, v_even_idx),
                                             v_tar.index_select(0, v_even_idx))
            else:
                vader_loss = 0
            loss += vader_loss
            loss_dict['v_pf'].append(vader_loss.item())
        if self.config['flair'] and un_freeze_fp:
            f_tar = torch.tensor(fp_batch['f_tar'], device=self.device)
            f_even_idx = get_even_class(f_tar, device=self.device)
            if f_even_idx is not None:
                flair_loss = F.cross_entropy(fp_result['flair'].index_select(0, f_even_idx),
                                             f_tar.index_select(0, f_even_idx))
            else:
                flair_loss = 0
            loss += flair_loss
            loss_dict['f_pf'].append(flair_loss.item())
        if self.config['sent'] and un_freeze_fp:
            s_tar = torch.tensor(fp_batch['s_tar'], device=self.device)
            s_even_idx = get_even_class(s_tar, device=self.device)
            if s_even_idx is not None:
                sent_loss = F.cross_entropy(fp_result['sent'].index_select(0, s_even_idx),
                                            s_tar.index_select(0, s_even_idx))
            else:
                sent_loss = 0
            loss += sent_loss
            loss_dict['s_pf'].append(sent_loss.item())
        if self.config['subj'] and un_freeze_fp:
            b_tar = torch.tensor(fp_batch['b_tar'], device=self.device)
            b_even_idx = get_even_class(b_tar, device=self.device)
            if b_even_idx is not None:
                subj_loss = F.cross_entropy(fp_result['subj'].index_select(0, b_even_idx),
                                            b_tar.index_select(0, b_even_idx))
            else:
                subj_loss = 0
            loss += subj_loss
            loss_dict['b_pf'].append(subj_loss.item())

        if self.config['emotion'] and un_freeze_fp:
            loss_per_dim = []
            e_tar = torch.tensor(fp_batch['e_tar'], device=self.device, dtype=torch.float)
            for dim_ in [0, 1, 2, 3, 4, 5]:
                pred = fp_result['emotion'][:, dim_]
                e_even_idx = get_even_class(e_tar[:, dim_], device=self.device)
                if e_even_idx is not None:
                    tmp_e_loss = F.binary_cross_entropy_with_logits(pred.index_select(0, e_even_idx),
                                                                    e_tar[:, dim_].index_select(0, e_even_idx))
                    loss_per_dim.append(tmp_e_loss)
            e_v_loss = sum(loss_per_dim) / len(loss_per_dim)
            loss += e_v_loss
            loss_dict['e_pf'].append(e_v_loss.item())
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
        if self.config['vader']:
            aux_targets.append('vader')
        if self.config['flair']:
            aux_targets.append('flair')
        if self.config['sent']:
            aux_targets.append('sent')
        if self.config['subj']:
            aux_targets.append('subj')
        if self.config['emotion']:
            aux_targets.append('emotion')
        if len(aux_targets) > 0:
            result = self.model(is_auxiliary=True,
                                x=a_batch['comment'],
                                target=aux_targets, device=self.device)
            if self.config['vader']:
                v_tar = torch.tensor(a_batch['v_tar'], device=self.device)
                v_even_idx = get_even_class(v_tar, device=self.device)
                a_v_loss = F.cross_entropy(result['vader'].index_select(0, v_even_idx),
                                           v_tar.index_select(0, v_even_idx))
                loss += a_v_loss
                loss_dict['v_ax'].append(a_v_loss.item())
            if self.config['flair']:
                f_tar = torch.tensor(a_batch['f_tar'], device=self.device)
                f_even_idx = get_even_class(f_tar, device=self.device)
                f_v_loss = F.cross_entropy(result['flair'].index_select(0, f_even_idx),
                                           f_tar.index_select(0, f_even_idx))
                loss += f_v_loss
                loss_dict['f_ax'].append(f_v_loss.item())
            if self.config['sent']:
                s_tar = torch.tensor(a_batch['s_tar'], device=self.device)
                s_even_idx = get_even_class(s_tar, device=self.device)
                s_v_loss = F.cross_entropy(result['sent'].index_select(0, s_even_idx),
                                           s_tar.index_select(0, s_even_idx))
                loss += s_v_loss
                loss_dict['s_ax'].append(s_v_loss.item())
            if self.config['subj']:
                b_tar = torch.tensor(a_batch['b_tar'], device=self.device)
                b_even_idx = get_even_class(b_tar, device=self.device)
                b_v_loss = F.cross_entropy(result['subj'].index_select(0, b_even_idx),
                                           b_tar.index_select(0, b_even_idx))
                loss += b_v_loss
                loss_dict['b_ax'].append(b_v_loss.item())

            if self.config['emotion']:
                loss_per_dim = []
                e_tar = torch.tensor(a_batch['e_tar'], device=self.device, dtype=torch.float)
                for dim_ in [0, 1, 2, 3, 4, 5]:
                    pred = result['emotion'][:, dim_]
                    e_even_idx = get_even_class(e_tar[:, dim_], device=self.device)
                    tmp_e_loss = F.binary_cross_entropy_with_logits(pred.index_select(0, e_even_idx),
                                                                    e_tar[:, dim_].index_select(0, e_even_idx))
                    loss_per_dim.append(tmp_e_loss)
                e_v_loss = sum(loss_per_dim) / 6.
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
            if not self.y_freq:
                for key, value in self.data_io.y_frequency.items():
                    self.y_freq[key] = torch.tensor(value, device=self.device, dtype=torch.float)

            for i, batch_idx in enumerate(train_iter):
                fp_batch, fp_result = self.get_result(batch_idx, train_examples)
                update_author = (self.config['free_fp'] >= e) and (self.config['freeze_aux'] <= e)
                update_fp = self.config['free_fp'] < e
                fp_loss = self.get_fp_loss(fp_result, fp_batch, self.train_loss,
                                           un_freeze_fp=update_fp,
                                           un_freeze_author=update_author)

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
                        np.mean(self.train_loss['author'][-self.config['check_step']:]),
                        np.mean(self.train_loss['t_ax'][-self.config['check_step']:]),
                        np.mean(self.train_loss['v_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['v_ax'][-self.config['check_step']:]),
                        np.mean(self.train_loss['f_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['f_ax'][-self.config['check_step']:]),
                        np.mean(self.train_loss['s_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['s_ax'][-self.config['check_step']:]),
                        np.mean(self.train_loss['b_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['b_ax'][-self.config['check_step']:]),
                        np.mean(self.train_loss['e_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['e_ax'][-self.config['check_step']:]))
                    print(train_log)
                    self.train_procedure.append(train_log)

                    if update_fp:
                        _, high_dev_perf, _ = self.get_perf(self.high_dev_iter, self.high_dev_examples)
                        # print('HIGH DEV: ', high_dev_perf)

                        if high_dev_perf['mean'] > self.dev_perf['mean']:
                            self.dev_perf = high_dev_perf
                            _, low_dev_perf, _ = self.get_perf(self.low_dev_iter, self.low_dev_examples)

                            torch.save({'model': self.model.state_dict(),
                                        'adam': self.sgd.state_dict()},
                                       os.path.join(self.config['root_folder'],
                                                    self.config['outlet'], 'best_model.pt'))
                            _, test_perf, test_preds = self.get_perf(self.test_iter, self.test_examples)
                            self.test_perf = test_perf

                            json.dump(test_perf, open(os.path.join(
                                self.config['root_folder'], self.config['outlet'], 'test_perf.json'), 'w'))
                            with open(os.path.join(self.config['root_folder'],
                                                   self.config['outlet'], 'test_pred.jsonl'), 'w') as f:
                                for pred in test_preds:
                                    data = json.dumps(pred)
                                    f.write(data)
                                    f.write('\n')

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
        tmp_pre_tar = {'vader': [[], []], 'flair': [[], []], 'sent': [[], []], 'subj': [[], []]}
        acc = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,
               'emotion': 0, 'mean': 0}
        self.model.eval()
        targets = []
        for i, batch_idx in enumerate(data_iter):
            fp_batch, fp_result = self.get_result(batch_idx, examples)

            for j, a in enumerate(fp_batch['author']):
                pred_record = {'vader': [0, 0], 'flair': [0, 0], 'sent': [0, 0], 'subj': [0, 0],
                               'emotion': [0, 0]}
                if self.config['vader']:
                    pred_record['vader'][0] = fp_result['vader'][j].argmax().item()
                    pred_record['vader'][1] = fp_batch['v_tar'][j]
                    tmp_cnt['vader'][int(pred_record['vader'][0] == pred_record['vader'][1])] += 1
                    tmp_pre_tar['vader'][0].append(pred_record['vader'][0])
                    tmp_pre_tar['vader'][1].append(pred_record['vader'][1])
                    targets.append('vader')
                if self.config['flair']:
                    pred_record['flair'][0] = fp_result['flair'][j].argmax().item()
                    pred_record['flair'][1] = fp_batch['f_tar'][j]
                    tmp_cnt['flair'][int(pred_record['flair'][0] == pred_record['flair'][1])] += 1
                    tmp_pre_tar['flair'][0].append(pred_record['flair'][0])
                    tmp_pre_tar['flair'][1].append(pred_record['flair'][1])
                    targets.append('flair')
                if self.config['sent']:
                    pred_record['sent'][0] = fp_result['sent'][j].argmax().item()
                    pred_record['sent'][1] = fp_batch['s_tar'][j]
                    tmp_cnt['sent'][int(pred_record['sent'][0] == pred_record['sent'][1])] += 1
                    tmp_pre_tar['sent'][0].append(pred_record['sent'][0])
                    tmp_pre_tar['sent'][1].append(pred_record['sent'][1])
                    targets.append('sent')
                if self.config['subj']:
                    pred_record['subj'][0] = fp_result['subj'][j].argmax().item()
                    pred_record['subj'][1] = fp_batch['b_tar'][j]
                    tmp_cnt['subj'][int(pred_record['subj'][0] == pred_record['subj'][1])] += 1
                    tmp_pre_tar['subj'][0].append(pred_record['subj'][0])
                    tmp_pre_tar['subj'][1].append(pred_record['subj'][1])
                    targets.append('subj')
                if self.config['emotion']:
                    pred_emo = [int(i > 0) for i in fp_result['emotion'][j].detach().tolist()]
                    pred_record['emotion'][0] = pred_emo
                    pred_record['emotion'][1] = fp_batch['e_tar'][j]
                    tmp_cnt['emotion'][0] += sum([pred == gold for pred, gold in zip(*pred_record['emotion'])])
                    tmp_cnt['emotion'][1] += sum([pred != gold for pred, gold in zip(*pred_record['emotion'])])
                pred_records.append(pred_record)
        self.model.train()
        tmp_acc_sum = []
        for key in targets:
            acc[key] = 1.0 * tmp_cnt[key][1] / (tmp_cnt[key][0] + tmp_cnt[key][1])
            tmp_acc_sum.append(acc[key])
        if self.config['emotion']:
            acc['emotion'] = 1.0 * tmp_cnt['emotion'][0] / (tmp_cnt['emotion'][0] + tmp_cnt['emotion'][1])
            tmp_acc_sum.append(acc['emotion'])
        acc['mean'] = sum(tmp_acc_sum) / len(tmp_acc_sum)
        f1, f1_each = {}, []
        for key, value in tmp_pre_tar.items():
            f1[key] = f1_score(tmp_pre_tar[key][1], tmp_pre_tar[key][0], labels=[1, 2], average='macro')
            f1_each.extend(f1[key] * 2)
        f1['mean'] = 1.0 * sum(f1_each) / (2 * len(f1_each))
        return acc, f1, pred_records


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


def get_even_class(tar, device):
    zero_idx = tar.eq(0).nonzero().view(-1)
    one_idx = tar.eq(1).nonzero().view(-1)
    two_idx = tar.eq(2).nonzero().view(-1)
    size_ = []
    if zero_idx.size(0) > 0 and zero_idx.size(0) > (tar.size(0) // 10):
        size_.append(zero_idx.size(0))
    if one_idx.size(0) > 0 and one_idx.size(0) > (tar.size(0) // 10):
        size_.append(one_idx.size(0))
    if two_idx.size(0) > 0 and two_idx.size(0) > (tar.size(0) // 10):
        size_.append(two_idx.size(0))

    if len(size_) > 1:
        min_size = min(size_)
    else:
        if zero_idx.size(0) > 0:
            size_.append(zero_idx.size(0))
        if one_idx.size(0) > 0:
            size_.append(one_idx.size(0))
        if two_idx.size(0) > 0:
            size_.append(two_idx.size(0))
        min_size = min(size_)

        if min_size == max(size_):
            return None

    result_idx = []
    if zero_idx.size(0) > 0:
        zero_perm = torch.randperm(zero_idx.size(0))
        perm_zero_idx = zero_idx[zero_perm][: min_size]
        result_idx.append(perm_zero_idx)
    if one_idx.size(0) > 0:
        one_perm = torch.randperm(one_idx.size(0))
        perm_one_idx = one_idx[one_perm][: min_size]
        result_idx.append(perm_one_idx)
    if two_idx.size(0) > 0:
        two_perm = torch.randperm(two_idx.size(0))
        perm_two_idx = two_idx[two_perm][: min_size]
        result_idx.append(perm_two_idx)

    if len(result_idx) > 1:
        return torch.cat(result_idx).to(device)
    else:
        return result_idx[0].to(device)


if __name__ == '__main__':
    print('test')
    print(np.mean([]))
