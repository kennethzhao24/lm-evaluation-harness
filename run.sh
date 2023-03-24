#!/bin/bash

# if [ -d "lm_cache" ]; then
#   rm -r lm_cache
# fi


python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_150k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_150k/opt_60000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_150k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_150k/opt_90000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_150k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_150k/opt_120000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte 
  

# python main.py \
#    --model gpt2 \
#    --batch_size 16 \
#    --model_args pretrained=/home/yzhao2/pretrained/GPT2/gpt_final.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,multirc,rte,record

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/pretrained/80M_new/config.json,pretrained=/home/yzhao2/pretrained/80M_new/opt_60000.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,multirc,rte,record

# python main.py \
#    --model opt \
#    --batch_size 8 \
#    --model_args config_file=/home/youpengzhao/code/pretrained/opt-125m/config.json,pretrained=/home/youpengzhao/code/pretrained/opt-125m/opt_final.pth \
#    --num_fewshot 1 \
#    --device cpu \
#    --tasks rte \
#    --quantization

# python main.py \
#    --model opt \
#    --model_args model_name=facebook/opt-125m \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks rte

