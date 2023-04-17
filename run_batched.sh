#!/bin/bash

if [ -d "lm_cache" ]; then
  rm -r lm_cache
fi

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_100m_512_300k/config.json,pretrained=/home/yzhao2/pretrained/opt_100m_512_300k/opt_60000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte,logiqa,pubmedqa

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_100m_512_300k/config.json,pretrained=/home/yzhao2/pretrained/opt_100m_512_300k/opt_120000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte,logiqa,pubmedqa

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_100m_512_300k/config.json,pretrained=/home/yzhao2/pretrained/opt_100m_512_300k/opt_180000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte,logiqa,pubmedqa

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_100m_512_300k/config.json,pretrained=/home/yzhao2/pretrained/opt_100m_512_300k/opt_240000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte,logiqa,pubmedqa

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_100m_512_300k/config.json,pretrained=/home/yzhao2/pretrained/opt_100m_512_300k/opt_300000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte,logiqa,pubmedqa
