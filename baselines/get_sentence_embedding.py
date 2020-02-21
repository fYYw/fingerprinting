import argparse
import collections
import logging
import json
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer
from transformers import BertModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    args = parser.parse_args()

