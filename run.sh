#!/bin/bash

if [ -d "lm_cache" ]; then
  rm -r lm_cache
fi

TASK=$1


python main.py \
   --model cerebras \
   --model_args model_name=cerebras/Cerebras-GPT-111M \
   --batch_size 16 \
   --num_fewshot 0 \
   --device 0 \
   --tasks $TASK


# python main.py \
#    --model opt \
#    --batch_size 8 \
#    --model_args model_name=facebook/opt-350m \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks $TASK


# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args model_name=facebook/opt-125m,pretrained=/home/yzhao2/pretrained/125M/opt_final.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_150k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_150k/opt_final.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte
