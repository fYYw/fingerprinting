import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import numpy as np


class ModelBase(nn.Module):
    def __init__(self, config):
        super(ModelBase, self).__init__()
        self.config = config
        self.token_embedding = None
        self.token_encoder = getattr(nn, config['rnn_type'].upper())(
            input_size=config['token_dim'], hidden_size=config['hid_dim'] // 2,
            num_layers=config['rnn_layer'], dropout=config['dropout'],
            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config['dropout'])

    def build_embedding(self, vocab=None, embedding=None):
        if vocab and embedding:
            weights = np.random.normal(loc=0, scale=0.1, size=(len(vocab), self.config['token_dim'])).astype(np.float32)
            for token, idx in vocab.items():
                if token in embedding:
                    weights[idx] = embedding[token]
            self.token_embedding = nn.Embedding.from_pretrained(torch.from_numpy(weights), freeze=False,
                                                                padding_idx=0)
        else:
            self.token_embedding = nn.Embedding(self.config['token_size'], self.config['token_dim'],
                                                padding_idx=0)

    def rnn_encode(self, rnn, x, mask, max_pool=True, mean_pool=True, last_pool=True):
        if len(x.size()) == 4:
            batch_size, seq_len, token_num, hid_dim = x.size()
            x = x.view(-1, token_num, hid_dim)
            mask = mask.view(-1, token_num)
            tensor = True
        else:
            batch_size, token_num, hid_dim = x.size()
            tensor = False

        x = self.dropout(x)
        token_len = mask.sum(-1)
        x_in = pack(x, token_len, batch_first=True, enforce_sorted=False)
        outputs, h_t = rnn(x_in)
        outputs, out_len = unpack(outputs, batch_first=True)

        if isinstance(h_t, tuple):
            h_t = h_t[0]
        h_t = h_t.transpose(0, 1).contiguous()  # batch, layer * num_direction, hid

        pooled = []
        if max_pool:
            tmp_output = outputs.masked_fill(mask.unsqueeze(-1).expand(-1, -1, outputs.size(-1)).eq(0),
                                             -float('inf'))  # batch, hid
            pooled.append(torch.max(tmp_output, dim=1)[0])
        if mean_pool:
            tmp_output = outputs.masked_fill(mask.unsqueeze(-1).expand(-1, -1, outputs.size(-1)).eq(0),
                                             0).sum(1) / token_len.unsqueeze(-1)  # batch, hid
            pooled.append(tmp_output)
        if last_pool:
            tmp_output = h_t.mean(1)  # batch, hid
            pooled.append(tmp_output)

        if len(pooled) > 1:
            pooled_result = torch.cat(pooled, dim=-1)
        else:
            pooled_result = pooled[0]
        if tensor:
            return pooled_result.view(batch_size, seq_len, -1)
        else:
            return pooled_result


class Model(ModelBase):
    def __init__(self, config):
        super(Model, self).__init__(config)

        author_final_dim = 0
        if config['build_author_emb']:
            self.author_embedding = nn.Embedding(config['author_size'], config['author_dim'])
            author_final_dim += config['author_dim']
        if config['build_author_track']:
            input_size = 2 * config['hid_dim'] * (int(config['token_max_pool']) +
                                                  int(config['token_mean_pool']))
            input_size += config['hid_dim'] * int(config['token_last_pool'])
            if config['build_sentiment_embedding']:
                if self.config['vader']:
                    self.vader_embed = nn.Embedding(3, config['sentiment_dim'])
                    input_size += config['sentiment_dim']
                if self.config['flair']:
                    self.flair_embed = nn.Embedding(3, config['sentiment_dim'])
                    input_size += config['sentiment_dim']
                if self.config['sent']:
                    self.sent_embed = nn.Embedding(3, config['sentiment_dim'])
                    input_size += config['sentiment_dim']
                if self.config['subj']:
                    self.subj_embed = nn.Embedding(3, config['sentiment_dim'])
                    input_size += config['sentiment_dim']
            if config['build_topic_predict'] and config['leverage_topic']:
                input_size += config['topic_size']
            self.timestamp_merge = nn.Linear(input_size, config['author_track_dim'])
            self.track_encoder = getattr(nn, config['rnn_type'].upper())(
                input_size=config['author_track_dim'], hidden_size=config['hid_dim'],
                num_layers=config['rnn_layer'], dropout=config['dropout'],
                batch_first=True, bidirectional=False)
            author_final_dim += config['hid_dim'] * (int(config['track_max_pool']) +
                                                     int(config['track_mean_pool']) +
                                                     int(config['track_last_pool']))

        if config['build_author_predict']:
            self.author_predict = nn.Linear(author_final_dim, config['author_size'])

        self.author_article_merge = nn.Sequential(
            nn.Linear(author_final_dim + (config['hid_dim'] // 2) * int(config['token_last_pool']) +
                      config['hid_dim'] * (int(config['token_max_pool']) +
                                           int(config['token_mean_pool'])),
                      config['hid_dim'] * 2),
            nn.Tanh(),
            nn.Linear(config['hid_dim'] * 2, config['hid_dim'] * 2),
            nn.Tanh())

        output_dim = 1 if config['loss_func'] == 'mse' else 3
        if config['vader']:
            self.vader = nn.Linear(config['hid_dim'] * 2, output_dim)
        if config['flair']:
            self.flair = nn.Linear(config['hid_dim'] * 2, output_dim)
        if config['sent']:
            self.sent = nn.Linear(config['hid_dim'] * 2, output_dim)
        if config['subj']:
            self.subj = nn.Linear(config['hid_dim'] * 2, output_dim)

        if config['build_topic_predict']:
            self.topic_predict = nn.Linear(config['hid_dim'] * (int(config['token_max_pool']) +
                                                                int(config['token_mean_pool'])) +
                                           (config['hid_dim'] // 2) * int(config['token_last_pool']),
                                           config['topic_size'])
        if config['vader']:
            self.vader_predict = nn.Linear(config['hid_dim'] * (int(config['token_max_pool']) +
                                                                int(config['token_mean_pool'])) +
                                           (config['hid_dim'] // 2) * int(config['token_last_pool']), 3)
        if config['flair']:
            self.flair_predict = nn.Linear(config['hid_dim'] * (int(config['token_max_pool']) +
                                                                int(config['token_mean_pool'])) +
                                           (config['hid_dim'] // 2) * int(config['token_last_pool']), 3)
        if config['sent']:
            self.sent_predict = nn.Linear(config['hid_dim'] * (int(config['token_max_pool']) +
                                                               int(config['token_mean_pool'])) +
                                          (config['hid_dim'] // 2) * int(config['token_last_pool']), 3)
        if config['subj']:
            self.subj_predict = nn.Linear(config['hid_dim'] * (int(config['token_max_pool']) +
                                                               int(config['token_mean_pool'])) +
                                          (config['hid_dim'] // 2) * int(config['token_last_pool']), 3)

    def fingerprint(self, author, read_target,
                    read_track, write_track, sentiment_track, device=torch.device('cpu')):
        """
        :param author: (batch, )
        :param read_target: (batch, token_len) * 2
        :param read_track: (batch, track_len, token_len), (batch, track_len, token_len), (batch, track_len)
        :param write_track: (batch, track_len, token_len), (batch, track_len, token_len), (batch, track_len)
        :param sentiment_track: (batch, track_len) * 4
        :param device: torch.device
        :return: dict
        """
        result = {}
        author = torch.tensor(author, device=device)
        # 0: tracks, 1: token_mask, 2: track_mask
        read_track = [torch.tensor(r, device=device) for r in read_track]
        write_track = [torch.tensor(w, device=device) for w in write_track]
        read_target = [torch.tensor(r, device=device) for r in read_target]

        article_target = self.rnn_encode(self.token_encoder,
                                         self.token_embedding(read_target[0]), read_target[1],
                                         max_pool=self.config['token_max_pool'],
                                         mean_pool=self.config['token_mean_pool'],
                                         last_pool=self.config['token_last_pool'])

        author_embeds = []
        if self.config['build_author_emb']:
            author_embeds.append(self.author_embedding(author))
        if self.config['build_author_track']:
            reads = self.rnn_encode(self.token_encoder,
                                    self.token_embedding(read_track[0]), read_track[1],
                                    max_pool=self.config['token_max_pool'],
                                    mean_pool=self.config['token_mean_pool'],
                                    last_pool=self.config['token_last_pool'])
            writes = self.rnn_encode(self.token_encoder,
                                     self.token_embedding(write_track[0]), write_track[1],
                                     max_pool=self.config['token_max_pool'],
                                     mean_pool=self.config['token_mean_pool'],
                                     last_pool=self.config['token_last_pool'])
            if self.config['detach_article']:
                tracks = [reads.detach(), writes.detach()]
            else:
                tracks = [reads, writes]

            track_embeds = torch.cat(tracks, dim=-1)
            track_embeds = torch.tanh(self.timestamp_merge(track_embeds))  # batch, track_len, dim
            tracks = self.rnn_encode(self.track_encoder, track_embeds, write_track[2],
                                     max_pool=self.config['track_max_pool'],
                                     mean_pool=self.config['track_mean_pool'],
                                     last_pool=self.config['track_last_pool'])
            author_embeds.append(tracks)

        if len(author_embeds) > 1:
            author_embeds = torch.cat(author_embeds, dim=-1)
        elif len(author_embeds) == 1:
            author_embeds = author_embeds[0]
        else:
            raise RuntimeError()

        final_rep = self.author_article_merge(torch.cat([author_embeds,
                                                         article_target], dim=-1))

        if self.config['flair']:
            result['flair'] = self.flair(final_rep)  # batch, seq, 3
        if self.config['vader']:
            result['vader'] = self.vader(final_rep)
        if self.config['sent']:
            result['sent'] = self.sent(final_rep)
        if self.config['subj']:
            result['subj'] = self.subj(final_rep)
        return result

    def auxiliary_predict(self, x, target, device):
        """
        :param x: articles: (batch, token_len) * 2
        :param target: str
        :param device: torch.device
        :return:
        """
        result = {}
        x = [torch.tensor(r, device=device) for r in x]
        embeds = self.rnn_encode(self.token_encoder,
                                 self.token_embedding(x[0]), x[1],
                                 max_pool=self.config['token_max_pool'],
                                 mean_pool=self.config['token_mean_pool'],
                                 last_pool=self.config['token_last_pool'])
        if 'vader' in target:
            result['vader'] = self.vader_predict(embeds)
        if 'flair' in target:
            result['flair'] = self.flair_predict(embeds)
        if 'sent' in target:
            result['sent'] = self.sent_predict(embeds)
        if 'subj' in target:
            result['subj'] = self.subj_predict(embeds)
        if 'topic' in target:
            result['topic'] = self.topic_predict(embeds)
        return result

    def forward(self, is_auxiliary, **kwargs):
        return self.fingerprint(author=kwargs['author'], read_target=kwargs['read_target'],
                                read_track=kwargs['read_track'], write_track=kwargs['write_track'],
                                sentiment_track=kwargs['sentiment_track'], device=kwargs['device'])


def sort_tensor_len(lengths):
    sorted_l, index = torch.sort(lengths, descending=True)
    _, track = torch.sort(index, descending=False)
    return sorted_l, index, track
