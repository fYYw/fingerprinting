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

    def get_bilinear_score(self, network, authors, articles):
        """ Authors: (batch, a_dim), articles: (batch, h_dim) """
        tmp = network(articles).unsqueeze(-1)  # batch, a_dim, 1
        return torch.bmm(authors.unsqueeze(1), tmp).squeeze(1).squeeze(1)  # batch, 1


class Model(ModelBase):
    def __init__(self, config):
        super(Model, self).__init__(config)

        hid_dim = (config['hid_dim'] // 2) * int(config['token_last_pool']) + config['hid_dim'] * (
                int(config['token_max_pool']) + int(config['token_mean_pool']))
        self.vader_predict = nn.Linear(hid_dim, 3)
        self.flair_predict = nn.Linear(hid_dim, 3)
        self.sent_predict = nn.Linear(hid_dim, 3)
        self.subj_predict = nn.Linear(hid_dim, 3)

    def predict(self, x, device):
        result = {}
        x = [torch.tensor(r, device=device) for r in x]
        embeds = self.rnn_encode(self.token_encoder,
                                 self.token_embedding(x[0]), x[1],
                                 max_pool=self.config['token_max_pool'],
                                 mean_pool=self.config['token_mean_pool'],
                                 last_pool=self.config['token_last_pool'])
        result['vader'] = self.vader_predict(embeds)
        result['flair'] = self.flair_predict(embeds)
        result['sent'] = self.sent_predict(embeds)
        result['subj'] = self.subj_predict(embeds)
        return result

    def forward(self, x, device):
        return self.predict(x, device)


def sort_tensor_len(lengths):
    sorted_l, index = torch.sort(lengths, descending=True)
    _, track = torch.sort(index, descending=False)
    return sorted_l, index, track
