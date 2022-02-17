# Step-unrolled Denoising Autoencoders for Text Generation

Unofficial PyTorch implementation of "Step-unrolled Denoising Autoencoders for
Text Generation"

> Work-in-progress, more experiments soon.

Currently implements unconditional character-level text generation on the text8
dataset.

Requires `x-transformers` and `ptpt` to be installed.

### Usage
Run SUNDAE training using config file `cfg.toml`:
```
python main.py --cfg-path cfg.toml
```

Sample text from trained model saved at `ckpt.pt`:
```
python main.py --cfg-path cfg.toml --resume ckpt.pt --sample --nb-samples 16
```

Other useful flags:
```
--seed          # set RNG seed 
--no-save       # disable saving of checkpoints [False]
--no-cuda       # disable the use of CUDA device [False]
--no-amp        # disable the use of automatic mixed precision [False]
--nb-workers    # set number of dataloader workers. [4]
```

### TODO:

- [ ] Add argmax-unrolled sampling
- [ ] Encoder-decoder translation experiments
- [ ] Pre-trained checkpoints

### Citation

**Step-unrolled Denoising Autoencoders for Text Generation**
> Nikolay Savinov, Junyoung Chung, Mikolaj Binkowski, Erich Elsen, Aaron van den Oord
```
@misc{savinov2021stepunrolled,
      title={Step-unrolled Denoising Autoencoders for Text Generation}, 
      author={Nikolay Savinov and Junyoung Chung and Mikolaj Binkowski and Erich Elsen and Aaron van den Oord},
      year={2021},
      eprint={2112.06749},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
