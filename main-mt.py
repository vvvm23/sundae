#!/usr/bin/env python
import argparse

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence

import toml
from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.log import debug, info, warning, error, critical
from ptpt.callbacks import CallbackType
from ptpt.utils import set_seed, get_parameter_count, get_device
from ptpt.wandb import WandbConfig

from data import get_de_en
from mtmodel import Transformer

def exists(val):
    return val is not None

def main(args):
    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    seed = set_seed(args.seed)

    info(f"random seed: {seed}")
    train_dataset, eval_dataset, de_tokenizer, en_tokenizer = get_de_en(**cfg.data)
    if cfg.src_lang == 'en' and cfg.tgt_lang == 'de':
        src_tokenizer = en_tokenizer
        tgt_tokenizer = de_tokenizer
    elif cfg.tgt_lang == 'en' and cfg.src_lang == 'de':
        src_tokenizer = de_tokenizer
        tgt_tokenizer = en_tokenizer
    else:
        raise ValueError('Unrecognized source-target language combination!')

    net = Transformer(**cfg.model)
    info(f"number of parameters: {get_parameter_count(net):,}")
    
    def get_random_text(shape):
        return torch.randint(cfg.model['tgt_num_tokens'], shape)

    def corrupt_text(batched_text):
        corruption_prob_per_sequence = torch.rand((batched_text.shape[0], 1))
        rand = torch.rand(batched_text.shape)
        mask = (rand < corruption_prob_per_sequence).to(batched_text.device)

        random_text = get_random_text(batched_text.shape).to(batched_text.device)
        return mask * random_text + ~mask * batched_text

    def logits_fn(net, src=None, src_len=None, tgt=None, tgt_len=None, src_mask=None, tgt_mask=None):
        tgt_len = tgt_len - 1
        src_len = src_len - 1
        corrupted_tgt = corrupt_text(tgt)
        all_logits = []
        src_emb = None
        batch_len_loss = None
        for _ in range(cfg.unroll_steps):
            tgt_logits, src_emb, len_loss = net(src, src_len, corrupted_tgt, tgt_len, src_mask=src_mask, tgt_mask=tgt_mask, src_h=src_emb)
            corrupted_tgt = Categorical(logits=tgt_logits).sample().detach()
            all_logits.append(tgt_logits)
            if exists(len_loss):
                batch_len_loss = len_loss
        final_logits = torch.cat(all_logits, axis=0)
        return final_logits, batch_len_loss

    # TODO: mask out loss?
    def loss_fn(net, batch):
        logits, len_loss = logits_fn(net, **batch)
        targets = batch['tgt'].repeat(cfg.unroll_steps, 1)
        accuracy = (logits.argmax(dim=-1) == targets).sum() / targets.numel()
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets) + len_loss
        return loss, accuracy*100., len_loss

    # TODO: can this be more efficient?
    def collate_fn(batch):
        def _pad_tensor(tensor, length, value):
            padded_tensor = torch.zeros(tensor.shape[0], length).to(tensor.dtype)
            padded_tensor.fill_(value)
            padded_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
            return padded_tensor

        src_len = torch.LongTensor([i[f'{cfg.src_lang}_len'] for i in batch])
        tgt_len = torch.LongTensor([i[f'{cfg.tgt_lang}_len'] for i in batch])


        src = pad_sequence([i[cfg.src_lang] for i in batch], batch_first=True, padding_value=src_tokenizer.pad_token_id)
        tgt = pad_sequence([i[cfg.tgt_lang] for i in batch], batch_first=True, padding_value=tgt_tokenizer.pad_token_id)

        src_mask = pad_sequence([i[f'{cfg.src_lang}_mask'] for i in batch], batch_first=True, padding_value=0).bool()
        tgt_mask = pad_sequence([i[f'{cfg.tgt_lang}_mask'] for i in batch], batch_first=True, padding_value=0).bool()

        return {
            'src': _pad_tensor(src, cfg.model['src_max_seq_len'], src_tokenizer.pad_token_id), 
            'tgt': _pad_tensor(tgt, cfg.model['tgt_max_seq_len'], tgt_tokenizer.pad_token_id), 
            'src_mask': _pad_tensor(src_mask, cfg.model['src_max_seq_len'], False), 
            'tgt_mask': _pad_tensor(tgt_mask, cfg.model['tgt_max_seq_len'], False), 
            'src_len': src_len, 'tgt_len': tgt_len,
        }
    
    # TODO: change for MT task
    @torch.inference_mode()
    def sample_fn(net):
        device = get_device(not args.no_cuda)

        batched_text = get_random_text((args.nb_samples, cfg.data['sequence_length'])).to(device)
        sample_mask = torch.zeros(args.nb_samples).bool().to(device)
        for n in range(cfg.sample['steps']):
            old_sample_mask = sample_mask.clone()
            logits = net(batched_text[~sample_mask])
            sample = Categorical(logits=logits / cfg.sample['temperature']).sample()
            
            mask = (torch.rand(sample.shape) > cfg.sample['sample_proportion']).to(batched_text.device)
            sample[mask] = batched_text[~sample_mask][mask]
 
            if n >= cfg.sample['min_steps']:
                sample_mask[~sample_mask] = torch.all((sample == batched_text[~sample_mask]).view(sample.shape[0], -1), dim=-1)

            if torch.all(sample_mask).item():
                break
            batched_text[~old_sample_mask] = sample
        debug(f"stopped sampling after {n+1} steps.")
        return batched_text

    trainer_cfg = TrainerConfig(
        **cfg.trainer,
        nb_workers = args.nb_workers,
        use_cuda = not args.no_cuda,
        use_amp = not args.no_amp,
        save_outputs = not args.no_save,
    )

    wandb_cfg = WandbConfig(
        project = 'sundae',
        entity = 'afmck',
        config = {'cfg': cfg, 'args': args}
    )

    trainer = Trainer(
        net = net,
        loss_fn = loss_fn,
        train_dataset = train_dataset,
        test_dataset = eval_dataset,
        collate_fn = collate_fn,
        cfg = trainer_cfg,
        wandb_cfg = wandb_cfg if args.wandb else None,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # TODO: change for MT task
    @torch.inference_mode()
    def callback_sample(trainer):
        trainer.net.eval()
        info("sampling from current model")
        samples = sample_fn(trainer.net)

        for i, sample in enumerate(samples):
            info(f"- " + ''.join(train_dataset.id_token[i.item()] for i in sample))
        trainer.net.train()

    if args.sample:
        callback_sample(trainer)
        exit()

    # trainer.register_callback(CallbackType.EvalEpoch, callback_sample, cfg.sample['sample_frequency'])
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default='config/wmt14ende.toml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--nb-workers', type=int, default=4)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--nb-samples', type=int, default=4)
    parser.add_argument('--min-steps', type=int, default=10)
    args = parser.parse_args()

    main(args)
