import torch
import datasets
from transformers import AutoTokenizer
import numpy as np

import random
from typing import Tuple
from collections import namedtuple
from string import ascii_lowercase
from pathlib import Path

class Text8Dataset(torch.utils.data.Dataset):
    def __init__(self,
            path: str,
            split: str = 'train',
            seq_len: int = 32,
            split_size: Tuple[int, int, int] = (90e6, 5e6, 5e6)
        ):
        super().__init__()
        self.seq_len = seq_len

        if split == 'train':
            seek_size = 0
            read_size = split_size[0]
        elif split == 'eval':
            seek_size = split_size[0]
            read_size = split_size[1]
        elif split == 'test':
            seek_size = sum(split_size[:2])
            read_size = split_size[2]

        with open(path, mode='r') as f:
            f.seek(int(seek_size))
            text = f.read(int(read_size))

        self.text = list(text)
        self.length = len(self.text)

        self.token_id, self.id_token = self._build_lookup()

    def _build_lookup(self):
        tokens = list(' ' + ascii_lowercase)
        return {t: i for i, t in enumerate(tokens)}, {i: t for i, t in enumerate(tokens)}

    def __getitem__(self, idx):
        sample = self.text[idx:idx+self.seq_len]
        
        if len(sample) < self.seq_len:
            sample = sample + [' ']*(self.seq_len - len(sample))

        return torch.LongTensor([self.token_id[t] for t in sample])

    def __len__(self):
        return self.length

def get_text8(path: str, seq_len: int = 32):
    return Text8Dataset(path=path, split='train', seq_len=seq_len),\
           Text8Dataset(path=path, split='eval', seq_len=seq_len),

def get_de_en(tokenizer_batch_size: int = 1000):
    dataset = datasets.load_dataset("wmt14", "de-en")

    if not Path('wmt14de-tokenizer').is_dir():
        print("> training new de tokenizer")
        base_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        corpus = (
            [d['de'] for d in dataset['train'][i:i+tokenizer_batch_size]['translation']]
            for i in range(0, len(dataset['train']), tokenizer_batch_size)
        )
        de_tokenizer = base_tokenizer.train_new_from_iterator(corpus, 32_000)
        de_tokenizer.save_pretrained("wmt14de-tokenizer")
    else:
        print("> loading de tokenizer from file")
        de_tokenizer = AutoTokenizer.from_pretrained('wmt14de-tokenizer')

    if not Path('wmt14en-tokenizer').is_dir():
        print("> training new en tokenizer")
        base_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        corpus = (
            [d['en'] for d in dataset['train'][i:i+tokenizer_batch_size]['translation']]
            for i in range(0, len(dataset['train']), tokenizer_batch_size)
        )
        en_tokenizer = base_tokenizer.train_new_from_iterator(corpus, 32_000)
        en_tokenizer.save_pretrained("wmt14en-tokenizer")
    else:
        print("> loading en tokenizer from file")
        en_tokenizer = AutoTokenizer.from_pretrained('wmt14en-tokenizer')

    return dataset['train'], dataset['validation'], de_tokenizer, en_tokenizer

def get_zh_en():
    dataset = load_dataset("wmt19", "zh-en")

if __name__ == '__main__':
    print(get_en_de())
