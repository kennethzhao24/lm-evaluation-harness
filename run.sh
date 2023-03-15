#!/bin/bash

python main.py \
   --model opt \
   --batch_size 8 \
   --model_args config_file=/home/youpengzhao/code/pretrained/opt-125m/config.json,pretrained=/home/youpengzhao/code/pretrained/opt-125m/opt_final.pth \
   --num_fewshot 1 \
   --device cpu \
   --tasks rte \
   --quantization

# python main.py \
#    --model opt \
#    --model_args model_name=facebook/opt-125m \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks rte

# python main.py \
#    --model opt \
#    --batch_size 8 \
#    --model_args config_file=/home/youpengzhao/code/pretrained/opt-125m/config.json,pretrained=/home/youpengzhao/code/pretrained/opt-125m/opt_final.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks rte


# python main.py \
#    --model opt \
#    --model_args config_file=/home/youpengzhao/code/lm-evaluation-harness/lm_eval/models/config.json,pretrained=/home/youpengzhao/code/opt_60000.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks record

   # --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,multirc,rte,record,wnli,qqp,cola,mnli,mnli_mismatched,mrpc,sst,qnli

# rm -r lm_cache

# python main.py \
#    --model opt \
#    --batch_size 8 \
#    --model_args config_file=/home/youpengzhao/code/pretrained/opt-125m/config.json,pretrained=/home/youpengzhao/code/pretrained/opt-125m/opt_final.pth \
#    --num_fewshot 0 \
#    --device cpu \
#    --tasks rte