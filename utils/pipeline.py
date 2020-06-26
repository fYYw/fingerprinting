import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score
from utils.misc_util import get_mse

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
        self.dev_examples, _ = data_io.build_eval_examples(
            split='dev', min_cnt=-1, pre_cnt=config['previous_comment_cnt'])
        self.dev_iter = data_io.build_iter_idx(self.dev_examples)
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
        self.train_log = "Epoch {0} Prog {1:.1f}% || v_tar: {2:.3f} || f_tar: {3:.3f} || s_tar: {4:.3f} || " \
                         "b_tar: {5:.3f} ||."
        self.train_loss = {'author': [0], 'v_pf': [0], 'f_pf': [0], 's_pf': [0], 'b_pf': [0], 'e_pf': [0],
                           't_ax': [0], 'v_ax': [0], 'f_ax': [0], 's_ax': [0], 'b_ax': [0], 'e_ax': [0]}

        self.dev_best_f1 = 0
        self.dev_perf_log = ''
        self.test_perf_log = ''
        self.train_procedure = []

    def get_fp_loss(self, fp_result, fp_batch, loss_dict):
        loss = 0

        if self.config['vader']:
            v_tar = torch.tensor(fp_batch['v_tar'], device=self.device)
            v_even_idx = get_even_class(v_tar, device=self.device)
            if v_even_idx is not None:
                if self.config['loss_func'] == 'ce':
                    vader_loss = F.cross_entropy(fp_result['vader'].index_select(0, v_even_idx),
                                                 v_tar.index_select(0, v_even_idx))
                else:
                    vader_loss = F.mse_loss(fp_result['vader'].index_select(0, v_even_idx),
                                            v_tar.index_select(0, v_even_idx).float() - 1)
            else:
                vader_loss = torch.zeros(1)
            loss = loss + vader_loss
            loss_dict['v_pf'].append(vader_loss.item())
        if self.config['flair']:
            f_tar = torch.tensor(fp_batch['f_tar'], device=self.device)
            f_even_idx = get_even_class(f_tar, device=self.device)
            if f_even_idx is not None:
                if self.config['loss_func'] == 'ce':
                    flair_loss = F.cross_entropy(fp_result['flair'].index_select(0, f_even_idx),
                                                 f_tar.index_select(0, f_even_idx))
                else:
                    flair_loss = F.mse_loss(fp_result['flair'].index_select(0, f_even_idx),
                                            f_tar.index_select(0, f_even_idx).float() - 1)
            else:
                flair_loss = torch.zeros(1)
            loss = loss + flair_loss
            loss_dict['f_pf'].append(flair_loss.item())
        if self.config['sent']:
            s_tar = torch.tensor(fp_batch['s_tar'], device=self.device)
            s_even_idx = get_even_class(s_tar, device=self.device)
            if s_even_idx is not None:
                if self.config['loss_func'] == 'ce':
                    sent_loss = F.cross_entropy(fp_result['sent'].index_select(0, s_even_idx),
                                                s_tar.index_select(0, s_even_idx))
                else:
                    sent_loss = F.mse_loss(fp_result['sent'].index_select(0, s_even_idx),
                                           s_tar.index_select(0, s_even_idx).float() - 1)
            else:
                sent_loss = torch.zeros(1)
            loss = loss + sent_loss
            loss_dict['s_pf'].append(sent_loss.item())
        if self.config['subj']:
            b_tar = torch.tensor(fp_batch['b_tar'], device=self.device)
            b_even_idx = get_even_class(b_tar, device=self.device)
            if b_even_idx is not None:
                if self.config['loss_func'] == 'ce':
                    subj_loss = F.cross_entropy(fp_result['vader'].index_select(0, b_even_idx),
                                                b_tar.index_select(0, b_even_idx))
                else:
                    subj_loss = F.mse_loss(fp_result['subj'].index_select(0, b_even_idx),
                                           b_tar.index_select(0, b_even_idx).float() - 1)
            else:
                subj_loss = torch.zeros(1)
            loss = loss + subj_loss
            loss_dict['b_pf'].append(subj_loss.item())
        return loss

    def get_result(self, batch_idx, examples):
        fp_batch = self.data_io.get_batch_fingerprint(batch_idx, examples)
        fp_result = self.model(author=fp_batch['author'],
                               read_target=fp_batch['r_target'],
                               read_track=fp_batch['r_track'],
                               write_track=fp_batch['w_track'],
                               sentiment_track=(fp_batch['v_tra'],
                                                fp_batch['f_tra'],
                                                fp_batch['s_tra'],
                                                fp_batch['b_tra']),
                               device=self.device)
        return fp_batch, fp_result

    def run(self):
        for e in range(self.epoch):
            grad_dict = {'layer': [], 'ave': [], 'max': []}
            train_examples = self.data_io.build_train_examples(pre_cnt=self.config['previous_comment_cnt'],
                                                               # min_cnt=self.config['min_comment_cnt'])
                                                               min_cnt=0)
            train_iter = self.data_io.build_iter_idx(train_examples, True)

            for i, batch_idx in enumerate(train_iter):
                fp_batch, fp_result = self.get_result(batch_idx, train_examples)
                fp_loss = self.get_fp_loss(fp_result, fp_batch, self.train_loss)

                loss = fp_loss
                try:
                    loss.backward()
                    if self.config['track_grad']:
                        track_grad(self.model.named_parameters(),
                                   grad_dict['layer'], grad_dict['ave'], grad_dict['max'])
                    if self.global_step % self.config['update_iter'] == 0:
                        clip_grad_norm_(self.params, self.config['grad_clip'])
                        self.sgd.step()
                        self.sgd.zero_grad()
                    self.global_step += 1
                except RuntimeError:
                    continue

                if self.global_step % (self.config['check_step']
                                       * self.config['update_iter']) == 0:
                    train_log = self.train_log.format(
                        e, 100.0 * i / len(train_iter),
                        np.mean(self.train_loss['author'][-self.config['check_step']:]),
                        np.mean(self.train_loss['v_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['f_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['s_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['b_pf'][-self.config['check_step']:]),
                        np.mean(self.train_loss['b_ax'][-self.config['check_step']:]))
                    print(train_log)
                    self.train_procedure.append(train_log)

                    f1_ave, perf_log, _ = self.get_perf(self.dev_iter, self.dev_examples)

                    if f1_ave > self.dev_best_f1:
                        self.dev_best_f1 = f1_ave
                        self.dev_perf_log = perf_log

                        torch.save({'model': self.model.state_dict(),
                                    'adam': self.sgd.state_dict()},
                                   os.path.join(self.config['root_folder'],
                                                self.config['outlet'], 'best_model.pt'))
                        _, test_perf, test_preds = self.get_perf(self.test_iter, self.test_examples)
                        self.test_perf_log = test_perf
                        # print(test_perf)
                        with open(os.path.join(self.config['root_folder'],
                                               self.config['outlet'], 'test_pred.txt'), 'w') as f:
                            for pred in test_preds:
                                if pred.endswith('\n'):
                                    f.write(pred)
                                else:
                                    f.write(pred + '\n')

            print('BEST DEV: ', self.dev_perf_log)
            print('BEST TEST: ', self.test_perf_log)
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

    def get_perf(self, data_iter, examples, labels=(-1, 0, 1)):
        results = {'author': [],
                   'aid': [],
                   'cid': [],
                   'vader': [[], [], []],
                   'flair': [[], [], []],
                   'sent': [[], [], []],
                   'subj': [[], [], []],
                   'mse': []}
        self.model.eval()
        for i, batch_idx in enumerate(data_iter):
            fp_batch, fp_result = self.get_result(batch_idx, examples)
            for j, author in enumerate(fp_batch['author']):
                results['author'].append(author)
                for y_p_name, y_t_name in zip(['vader', 'flair', 'sent', 'subj'],
                                              ['v_tar', 'f_tar', 's_tar', 'b_tar']):
                    tar_value = fp_batch[y_t_name][j]
                    pred_label = fp_result[y_p_name][j].argmax().item()
                    results[y_p_name][0].append(pred_label)
                    results[y_p_name][1].append(tar_value)
        self.model.train()
        perf_log = ''
        mean_f1 = []
        for y_name in ['vader', 'flair', 'sent', 'subj']:
            preds, tars = results[y_name][0], results[y_name][1]
            f1_ = f1_score(tars, preds, labels=[0, 2], average='macro')
            perf_log += "{0} f1: {1:.4f}, mse: {2:.4f}; ".format(y_name, f1_, 0.0)
            mean_f1.append(f1_)
        logs = [perf_log + '\n']
        packed_result = zip(results['author'],
                            results['vader'][0], results['vader'][1],
                            results['flair'][0], results['flair'][1],
                            results['sent'][0], results['sent'][1],
                            results['subj'][0], results['subj'][1])
        for au, vp, vt, fp, ft, sp, st, bp, bt in packed_result:
            logs.append('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(au, vp, vt,
                                                                      fp, ft, sp, st, bp, bt))
        return 1.0 * sum(mean_f1) / len(mean_f1), perf_log, logs


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
    a = [1, 0, 1, 0, 2, 2, 1]
    b = [0, 1, 2, 2, 2, 1, 1]
    f = f1_score(a, b, average='macro')
    f1 = f1_score(a, b, average=None)
    print(f1)
    print(f)
