import argparse

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import toml
from types import SimpleNamespace
from pathlib import Path

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.log import debug, info, warning, error, critical
from ptpt.callbacks import CallbackType
from ptpt.utils import set_seed, get_parameter_count, get_device

from data import get_text8

from x_transformers import TransformerWrapper, Encoder

def main(args):
    seed = set_seed(args.seed)
    info(f"random seed: {seed}")
    train_dataset, eval_dataset = get_text8('data/text8', seq_len=32)
    vocab_size = len(train_dataset.token_id)
    unroll_steps = 3

    net = TransformerWrapper(
        num_tokens = vocab_size,
        max_seq_len = 32,
        attn_layers = Encoder(
            dim = 512,
            depth = 12,
            head = 8,
            use_scalenorm = True,
            ff_glu = True,
            rotary_pos_emb=True,
        )
    )
    info(f"number of parameters: {get_parameter_count(net):,}")
    
    def get_random_text(shape, vocab_size):
        return torch.randint(vocab_size, shape)

    def corrupt_text(batched_text, vocab_size):
        corruption_prob_per_sequence = torch.rand((batched_text.shape[0], 1))
        rand = torch.rand(batched_text.shape)
        mask = (rand < corruption_prob_per_sequence).to(batched_text.device)

        random_text = get_random_text(batched_text.shape, vocab_size).to(batched_text.device)
        return mask * random_text + ~mask * batched_text

    def logits_fn(net, batched_text):
        samples = corrupt_text(batched_text, vocab_size)
        all_logits = []
        for _ in range(unroll_steps):
            logits = net(samples)
            samples = Categorical(logits=logits).sample().detach()
            all_logits.append(logits)
        final_logits = torch.cat(all_logits, axis=0)
        return final_logits

    def loss_fn(net, batched_text):
        logits = logits_fn(net, batched_text)
        targets = batched_text.repeat(unroll_steps, 1)
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
        return loss,
    
    @torch.inference_mode()
    def sample_fn(net, steps, nb_samples, seq_len, vocab_size, temperature, sample_proportion):
        device = get_device(not args.no_cuda)

        batched_text = get_random_text((nb_samples, seq_len), vocab_size).to(device)
        for _ in range(steps):
            logits = net(batched_text)
            sample = Categorical(logits=logits/temperature).sample()
            
            mask = (torch.rand(batched_text.shape) > sample_proportion).to(batched_text.device)
            sample[mask] = batched_text[mask]

            if torch.equal(sample, batched_text):
                break
            batched_text = sample
        return batched_text
    
    trainer_cfg = TrainerConfig(
        exp_name = 'text8-sundae',
        exp_dir = 'exp',
        batch_size = 200,
        nb_batches = (100, 10),
        learning_rate = 1e-5,
        nb_workers = args.nb_workers,
        use_cuda = not args.no_cuda,
        use_amp = not args.no_amp,
        save_outputs = not args.no_save,
    )

    trainer = Trainer(
        net = net,
        loss_fn = loss_fn,
        train_dataset = train_dataset,
        test_dataset = eval_dataset,
        cfg = trainer_cfg,
    )
    
    def callback_sample(trainer):
        trainer.net.eval()
        info("sampling from current model")
        samples = sample_fn(
            trainer.net,
            steps=1000,
            nb_samples=4,
            seq_len=32,
            vocab_size=vocab_size,
            temperature=0.8,
            sample_proportion=0.3,
        )

        for i, sample in enumerate(samples):
            info(f"- " + ''.join(train_dataset.id_token[i.item()] for i in sample))

        trainer.net.train()
    trainer.register_callback(CallbackType.EvalEpoch, callback_sample)

    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default='config/cifar10.toml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--nb-workers', type=int, default=4)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()

    main(args)
