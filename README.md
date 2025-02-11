# Language Model Evaluation Harness

## Overview

This project provides a unified framework to test autoregressive language models (GPT-2, GPT-3, GPTNeo, etc) on a large number of different evaluation tasks.

Features:
- 200+ tasks implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for GPT-2, GPT-3, GPT-Neo, GPT-NeoX, and GPT-J, with flexible tokenization-agnostic interface.
- Task versioning to ensure reproducibility.

### Changes to original repo
- [x] Add support for OPT Model
- [x] Add quantization for OPT Model


## Install

```
git clone https://github.com/kennethzhao24/lm-evaluation-harness
cd lm-evaluation-harness
pip install -r requirements.txt
```

## Basic Usage
The following command loads pretrained weights from official OPT-125M model and evaluate it on RTE task in zero-shot setting:
```bash
python main.py \
   --model opt \
   --model_args model_name=facebook/opt-125m \
   --num_fewshot 0 \
   --device 0 \
   --tasks rte \
   --output_path ./results/opt_125m_zero_shot.json
```

The following command loads pretrained weights from your own pretrained OPT-125M model and evaluate it on RTE task in zero-shot setting:

```bash
python main.py \
   --model opt \
   --batch_size 8 \
   --model_args config_file=/home/youpengzhao/code/pretrained/opt-125m/config.json,pretrained=/home/youpengzhao/code/pretrained/opt-125m/opt_final.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks rte \
   --output_path ./results/opt_125m_zero_shot.json
```
8-bit Quantization is also supported (only on CPU):
```bash
python main.py \
   --model opt \
   --batch_size 8 \
   --model_args config_file=/home/youpengzhao/code/pretrained/opt-125m/config.json,pretrained=/home/youpengzhao/code/pretrained/opt-125m/opt_final.pth \
   --num_fewshot 0 \
   --device cpu \
   --tasks rte \
   --quantization
```
For GPT-2 eval
```bash
python main.py \
   --model gpt2 \
   --model_args pretrained=/home/youpengzhao/code/pretrained/GPT2/gpt_final.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks rte
```

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
