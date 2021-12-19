import torch
import numpy as np

import random
from typing import Tuple
from collections import namedtuple
from string import ascii_lowercase

# TODO: optimize?
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

if __name__ == '__main__':
    from tqdm import tqdm
    dataset = Text8Dataset('data/text8')
    x = dataset.__getitem__(len(dataset) - 2)
    print(x)

    # for i in tqdm(range(len(dataset))):
        # assert len(dataset.__getitem__(i)) == 32
