import os
import re
import time
import json
import collections
import unicodedata
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer
from transformers import BertModel


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b, index):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.index = index


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, index):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.index = index


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            print("*** Example ***")
            print("unique_id: %s" % (example.unique_id))
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        if ex_index % 10000 == 0:
            print("convert input: %s" % (ex_index))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                index=example.index))
    return features


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip().split('\t')
            index = line[0]
            text = line[1]
            examples.append(
                InputExample(unique_id=unique_id, text_a=text, text_b=None, index=index))
            unique_id += 1
    return examples


def build_bert_input(folder):
    a_tsv = os.path.join(folder, 'article_idx_text.tsv')
    c_tsv = os.path.join(folder, 'comment_idx_text.tsv')
    comment_idx = json.load(open(os.path.join(folder, 'comment_idx.json')))
    aid_cid_dict = {}
    for cid, value in comment_idx.items():
        aid = value['aid']
        if aid not in aid_cid_dict:
            aid_cid_dict[aid] = set()
        aid_cid_dict[aid].add(cid)

    ac_dict = json.load(open(os.path.join(folder, 'article_comment.json')))
    with open(a_tsv, 'w', encoding='utf-8') as af, open(c_tsv, 'w', encoding='utf-8') as cf:
        for aid, value in ac_dict.items():
            article_dict = value['article']
            comment_dict = value['comment']
            if aid in aid_cid_dict:
                a_content = unicodedata.normalize('NFD', article_dict['content'])
                a_title = unicodedata.normalize('NFD', article_dict['title'])
                a_content = a_content if a_content else a_title
                af.write('{}\t{}\n'.format(aid, a_content))

                for cid, c_value in comment_dict.items():
                    if cid in aid_cid_dict[aid]:
                        c_content = unicodedata.normalize('NFD', c_value['content'])
                        tmp = re.sub(r'[^a-zA-Z0-9.?!]', '', c_content)
                        if c_content and tmp:
                            cf.write('{}\t{}\n'.format(cid, c_content))
    del ac_dict, comment_idx


def get_bert_embeds(input_file, output_file, bert_model, tokenizer, device, max_seq_length=256, batch_size=64):
    examples = read_examples(input_file)
    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    sentence_embeddings = []
    bert_model.eval()
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="testing"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            _, pooled_output = bert_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        sentence_embeddings.extend(pooled_output.detach().cpu().numpy())
    all_index = [f.index for f in features]
    print(len(sentence_embeddings[0]))

    instances = {}
    for i in range(len(sentence_embeddings)):
        index = all_index[i]
        embedding = sentence_embeddings[i]

        if index not in instances:
            instances[index] = {}
            instances[index]['text'] = embedding

    fout = open(output_file, 'w')
    for instance in instances.items():
        output_json = collections.OrderedDict()
        output_json['index'] = instance[0]
        output_json['text'] = [round(x.item(), 8) for x in instance[1]['text']]
        fout.write(json.dumps(output_json) + '\r\n')
    fout.close()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for outlet in [
        'Archiveis',
        'DailyMail',
        'foxnews',
        'NewYorkTimes',
        'theguardian',
        'wsj'
    ]:
        build_bert_input(os.path.join('d:/data/outlets', outlet))
        in_article = os.path.join('d:/data/outlets', outlet, 'article_idx_text.tsv')
        out_article = os.path.join('d:/data/outlets', outlet, 'article_bert.tsv')
        get_bert_embeds(in_article, out_article, model, tokenizer,
                        max_seq_length=256, batch_size=64, device=device)
        in_comment = os.path.join('d:/data/outlets', outlet, 'comment_idx_text.tsv')
        out_comment = os.path.join('d:/data/outlets', outlet, 'comment_bert.tsv')
        get_bert_embeds(in_comment, out_comment, model, tokenizer,
                        max_seq_length=256, batch_size=64, device=device)
