#!/bin/bash

if [ -d "lm_cache" ]; then
  rm -r lm_cache
fi

python main.py \
   --model gpt2 \
   --batch_size 16 \
   --num_fewshot 0 \
   --device 0 \
   --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte,logiqa,pubmedqa,qa4mre_2013

python main.py \
   --model gpt2 \
   --batch_size 16 \
   --model_args pretrained=gpt2-medium \
   --num_fewshot 0 \
   --device 0 \
   --tasks $TASK

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args model_name=facebook/opt-125m \
   --num_fewshot 0 \
   --device 0 \
   --tasks qa4mre_2013

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args model_name=facebook/opt-350m \
   --num_fewshot 0 \
   --device 0 \
   --tasks qa4mre_2013

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_300k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_300k/opt_final.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks logiqa,pubmedqa,qa4mre_2013

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_150k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_150k/opt_final.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks qa4mre_2013

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/opt_final.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks qa4mre_2013





python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_60m_512_100k/config.json,pretrained=/home/yzhao2/pretrained/opt_60m_512_100k/opt_final.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks qa4mre_2013

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_60m_512_100k_ws/config.json,pretrained=/home/yzhao2/pretrained/opt_60m_512_100k_ws/opt_final.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks qa4mre_2013

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_80m_256_60k/config.json,pretrained=/home/yzhao2/pretrained/opt_80m_256_60k/opt_final.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks qa4mre_2013

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_80m_512_100k/config.json,pretrained=/home/yzhao2/pretrained/opt_80m_512_100k/opt_100000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks qa4mre_2013

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_80m_512_300k/config.json,pretrained=/home/yzhao2/pretrained/opt_80m_512_300k/opt_300000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks qa4mre_2013

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/opt_80m_512_300k_ws/config.json,pretrained=/home/yzhao2/pretrained/opt_80m_512_300k_ws/opt_300000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks qa4mre_2013