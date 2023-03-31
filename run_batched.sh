#!/bin/bash

if [ -d "lm_cache" ]; then
  rm -r lm_cache
fi

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/youpengzhao/code/pretrained/60M_150k/config.json,pretrained=/home/youpengzhao/code/pretrained/60M_150k/opt_120000.pth \
#    --num_fewshot 0 \
#    --device cpu \
#    --quantization \
#    --tasks rte

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/youpengzhao/code/pretrained/60M_150k/config.json,pretrained=/home/youpengzhao/code/pretrained/60M_150k/opt_120000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks rte

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_300k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_60m_300k/opt_final.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte


# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/opt_60000.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/opt_90000.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/opt_120000.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/opt_150000.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/config.json,pretrained=/home/yzhao2/Entropy-Transformer-Design/results_pile/zad_100m_200k/opt_180000.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

python main.py \
   --model opt \
   --batch_size 16 \
   --model_args config_file=/home/yzhao2/pretrained/80M_300K/config.json,pretrained=/home/yzhao2/pretrained/80M_300K/opt_60000.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/pretrained/60M/config.json,pretrained=/home/yzhao2/pretrained/60M/opt_final.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/pretrained/60M_WS/config.json,pretrained=/home/yzhao2/pretrained/60M_WS/opt_60000.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte

# python main.py \
#    --model opt \
#    --batch_size 16 \
#    --model_args config_file=/home/yzhao2/pretrained/80M_new/config.json,pretrained=/home/yzhao2/pretrained/80M_new/opt_60000.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,rte