import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import numpy as np


class Model(nn.Module):
    def __init__(self,
                 config):
        super(Model, self).__init__()
        self.token_embedding = None
        self.token_encoder = getattr(nn, config['rnn_type'].upper())(
            input_size=config['token_dim'], hidden_size=config['hid_dim'] // 2,
            num_layers=1, dropout=config['dropout'],
            batch_first=True, bidirectional=True)

        author_final_dim = 0
        if config['build_author_emb']:
            self.author_embedding = nn.Embedding(config['author_size'], config['author_dim'])
            author_final_dim += config['author_dim']

        if config['build_author_track']:
            input_size = 2 * config['hid_dim']
            if config['build_sentiment_embedding']:
                input_size += 4 * config['sentiment_dim']
            if config['build_topic_predict'] and config['leverage_topic']:
                input_size += config['topic_size']
            if config['leverage_emotion']:
                input_size += 6

            self.timestamp_merge = nn.Sequential(
                nn.Linear(input_size, config['author_track_dim']),
                nn.ReLU(),
                nn.Linear(config['author_track_dim'], config['author_track_dim']),
                nn.ReLU())

            self.track_encoder = getattr(nn, config['rnn_type'].upper())(
                input_size=config['author_track_dim'], hidden_size=config['author_track_dim'],
                num_layers=config['rnn_layer'], dropout=config['dropout'],
                batch_first=True, bidirectional=False)
            author_final_dim += config['author_track_dim'] * config['rnn_layer']
            # self.author_merge = nn.Linear(config['hid_dim'] * 2, config['author_dim'])

        if config['build_author_predict']:
            self.author_predict = nn.Linear(author_final_dim, config['author_size'])

        if config['build_topic_predict']:
            self.topic_predict = nn.Linear(config['hid_dim'], config['topic_size'])

        in_dim = author_final_dim + config['hid_dim']
        self.dropout = nn.Dropout(config['dropout'])

        if config['sentiment_fingerprinting']:
            self.vader_predict = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))
            self.flair_predict = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))
            self.blob_sent = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))
            self.blob_subj = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))

        if config['emotion_fingerprinting']:
            self.emotion_predict = nn.Linear(in_dim, config['emotion_dim'])

        if config['build_sentiment_embedding']:
            self.vader_embed = nn.Embedding(3, config['sentiment_dim'])
            self.vader_embed.weight = self.vader_predict[2].weight
            self.flair_embed = nn.Embedding(3, config['sentiment_dim'])
            self.flair_embed.weight = self.flair_predict[2].weight
            self.sent_embed = nn.Embedding(3, config['sentiment_dim'])
            self.sent_embed.weight = self.blob_sent[2].weight
            self.subj_embed = nn.Embedding(3, config['sentiment_dim'])
            self.subj_embed.weight = self.blob_subj[2].weight

        self.config = config

    def _encode_seq_seq_(self, seq_seq_tensor, token_mask):
        batch_size, seq_len, token_size = seq_seq_tensor.size()
        token_encode_mtx = seq_seq_tensor.view(-1, token_size)
        token_len = token_mask.view(-1, token_size).sum(-1)
        # token_sorted_len, token_order, token_track = sort_tensor_len(token_len)

        embeds, ht = self._rnn_encode_(self.token_encoder, self.token_embedding(token_encode_mtx),
                                       token_len)
        embeds = embeds.view(batch_size, seq_len, token_size, embeds.size(-1))
        ht = ht.view(batch_size, seq_len, ht.size(-1))
        return embeds, ht

    def forward(self, author, read_track, write_track, article_pack, sentiments, emotion, device, train=True):
        """
        :param author:
        :param read_track:
        :param write_track:
        :param article_pack:
        :param device:
        :param sentiments: list of tensors
        :param emotion
        :param train
        :return: result:
        """
        result = {}
        author = torch.tensor(author, device=device)
        read_track = [torch.tensor(r, device=device) for r in read_track]
        write_track = [torch.tensor(w, device=device) for w in write_track]
        seq_len = write_track[2].view(-1, write_track[0].size(1)).sum(-1)

        r_embeds, r_ht = self._encode_seq_seq_(read_track[0], read_track[1])

        author_embeds = []
        if self.config['build_author_track']:
            w_embeds, w_ht = self._encode_seq_seq_(write_track[0], write_track[1])

            tracks = [r_ht, w_ht]
            if self.config['build_sentiment_embedding']:
                tracks.extend([self.vader_embed(torch.tensor(sentiments[0][0], device=device)),
                               self.flair_embed(torch.tensor(sentiments[1][0], device=device)),
                               self.sent_embed(torch.tensor(sentiments[2][0], device=device)),
                               self.subj_embed(torch.tensor(sentiments[3][0], device=device))])
            if self.config['build_topic_predict'] and self.config['leverage_topic']:
                predict_topic = F.softmax(self.topic_predict(r_ht).detach(), dim=-1)
                tracks.append(predict_topic)
            if self.config['leverage_emotion']:
                tracks.append(torch.tensor(emotion[0], device=device, dtype=torch.float))
            track_embeds = torch.cat(tracks, dim=-1)[:, :-1, :]
            track_embeds = F.relu(self.timestamp_merge(track_embeds))
            # seq_sorted_len, seq_order, seq_track = sort_tensor_len(seq_len)
            _, track_ht = self._rnn_encode_(self.track_encoder, track_embeds,
                                            seq_len - 1)
            author_embeds.append(track_ht)
            # author_embeds = self.author_merge(track_embeds)
            # author_embeds = self.author_merge(track_embeds[:, :-1, :])

        if self.config['build_author_emb']:
            # author_embeds = self.author_embedding(author).unsqueeze(1)  # batch, 1, a_hid
            # author_embeds = author_embeds.expand(-1, r_embeds.size(1), -1)
            author_embeds.append(self.author_embedding(author))

        if len(author_embeds) > 1:
            author_embeds = torch.cat(author_embeds, dim=-1)
        elif len(author_embeds) == 1:
            author_embeds = author_embeds[0]
        else:
            raise NotImplementedError()

        final_idx = (seq_len - 1).view(-1, 1).expand(-1, r_ht.size(2))
        final_idx = final_idx.unsqueeze(1)
        final_rt = r_ht.gather(1, final_idx).squeeze(1)
        # final_rt = r_ht[:, -1, :]  # last time stamp to predict
        if self.config['sentiment_fingerprinting']:
            result['flair'] = self.flair_predict(torch.cat((author_embeds, final_rt), dim=-1))  # batch, seq, 3
            result['vader'] = self.vader_predict(torch.cat((author_embeds, final_rt), dim=-1))
            result['sent'] = self.blob_sent(torch.cat((author_embeds, final_rt), dim=-1))
            result['subj'] = self.blob_subj(torch.cat((author_embeds, final_rt), dim=-1))

        if self.config['emotion_fingerprinting']:
            result['emotion'] = self.emotion_predict(torch.cat((author_embeds, final_rt), dim=-1))

        if self.config['build_author_predict'] and train:
            result['author'] = self.author_predict(author_embeds)

        if self.config['build_topic_predict'] and train:
            article_pack = torch.tensor(article_pack[0], device=device)
            batch_article_pack_size, _ = article_pack.size()
            token_len = article_pack.ne(0).sum(-1)
            token_sorted_len, token_order, token_track = sort_tensor_len(token_len)
            _, ht = self._rnn_encode_(self.token_encoder, self.token_embedding(article_pack),
                                      token_sorted_len, token_order, token_track)  # _, ht: (batch, hid_dim)
            result['topic'] = self.topic_predict(ht)
        return result

    def _rnn_encode_(self, rnn, x, length, order=None, track=None):
        if len(x.size()) == 3:
            batch_size, seq_len, token_num = x.size()
        elif len(x.size()) == 2:
            batch_size, token_num = x.size()
        else:
            raise NotImplementedError("Not support input dimensions {}".format(x.size()))

        if order is not None:
            x = x.index_select(0, order)
        x = self.dropout(x)
        x = pack(x, length, batch_first=True, enforce_sorted=False)
        outputs, h_t = rnn(x)
        outputs = unpack(outputs, batch_first=True)[0]
        if isinstance(h_t, tuple):
            h_t = h_t[0]
        if track is not None:
            outputs = outputs[track]
            h_t = h_t.index_select(1, track).transpose(0, 1).contiguous()
        else:
            h_t = h_t.transpose(0, 1).contiguous()
        return outputs, h_t.view(batch_size, -1)

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


def sort_tensor_len(lengths):
    sorted_l, index = torch.sort(lengths, descending=True)
    _, track = torch.sort(index, descending=False)
    return sorted_l, index, track
