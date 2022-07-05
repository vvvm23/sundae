#!/usr/bin/env python
import argparse

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence

import random
import toml
from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path

from ptpt.utils import set_seed, get_parameter_count, get_device
from ptpt.log import debug, info, warning, error, critical

from data import get_de_en
from mtmodel import Transformer

def main(args):
    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    device = get_device(args.no_cuda)
    seed = set_seed(args.seed)

    info(f"random seed: {seed}")
    train_dataset, eval_dataset, de_tokenizer, en_tokenizer = get_de_en(**cfg.data)
    if cfg.src_lang == 'en' and cfg.tgt_lang == 'de':
        src_tokenizer = en_tokenizer
        tgt_tokenizer = de_tokenizer
    elif cfg.tgt_lang == 'en' and cfg.src_lang == 'de':
        c_tokenizer = de_tokenizer
        tgt_tokenizer = en_tokenizer
    else:
        raise ValueError('Unrecognized source-target language combination!')

    net = Transformer(**cfg.model)
    info(f"number of parameters: {get_parameter_count(net):,}")

    def get_random_text(shape):
        return torch.randint(cfg.model['tgt_num_tokens'], shape)

    def sample_fn(net, src=None, src_len=None, src_mask=None, 
            temperature = 0.3, sample_proportion = 0.3,
            min_steps = 10, max_steps=10
        ):
        src_len = src_len - 1
        corrupted_tgt = get_random_text(src.shape).to(device)
        src_emb = None
        for _ in tqdm(range(max_steps)):
            tgt_logits, src_emb = net.sample_step(src, src_len, corrupted_tgt, src_mask=src_mask, src_h=src_emb)
            sample = Categorical(logits=tgt_logits / temperature).sample().detach()
            mask = (torch.rand(sample.shape) > sample_proportion).to(corrupted_tgt.device)
            sample[mask] = corrupted_tgt[mask]
            corrupted_tgt = sample

            print(tgt_tokenizer.batch_decode(corrupted_tgt, skip_special_tokens=True)[0])

        return corrupted_tgt

    def pad_tensor(tensor, length, value):
        padded_tensor = torch.zeros(tensor.shape[0], length).to(tensor.dtype)
        padded_tensor.fill_(value)
        padded_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
        return padded_tensor

    net = net.to(device)
    if args.resume:
        net.load_state_dict(torch.load(args.resume)['net'])
    net.eval()

    random_batch = eval_dataset.__getitem__(random.randint(0, len(eval_dataset) - 1))

    src = random_batch[cfg.src_lang].unsqueeze(0).to(device)
    src = pad_tensor(src, cfg.model['src_max_seq_len'], src_tokenizer.pad_token_id)
    src_len = random_batch[f'{cfg.src_lang}_len'].to(device)

    tgt = random_batch[cfg.tgt_lang].unsqueeze(0).to(device)
    tgt = pad_tensor(tgt, cfg.model['tgt_max_seq_len'], tgt_tokenizer.pad_token_id)
    tgt_len = random_batch[f'{cfg.tgt_lang}_len'].to(device)


    src_mask = random_batch[f'{cfg.src_lang}_mask'].unsqueeze(0).to(device)
    src_mask = pad_tensor(src_mask, cfg.model['src_max_seq_len'], False).bool()
    tgt_mask = random_batch[f'{cfg.tgt_lang}_mask'].unsqueeze(0).to(device)
    tgt_mask = pad_tensor(tgt_mask, cfg.model['tgt_max_seq_len'], False).bool()

    predicted_tgt = sample_fn(net, src, src_len, src_mask)

    print("Source:", src_tokenizer.batch_decode(src, skip_special_tokens=True)[0])
    print()
    print("Target:", tgt_tokenizer.batch_decode(tgt, skip_special_tokens=True)[0])
    print()
    print("Predicted:", tgt_tokenizer.batch_decode(predicted_tgt, skip_special_tokens=True)[0])
    exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default='config/wmt14ende.toml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--min-steps', type=int, default=10)
    args = parser.parse_args()

    main(args)
