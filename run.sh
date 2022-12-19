python main.py \
   --model opt \
   --model_args model_name=facebook/opt-125m,pretrained=/home/youpengzhao/code/test_opt.pth \
   --num_fewshot 0 \
   --device 0 \
   --tasks wnli


# python main.py \
#    --model opt \
#    --model_args config_file=/home/youpengzhao/code/lm-evaluation-harness/lm_eval/models/config.json,pretrained=/home/youpengzhao/code/opt_60000.pth \
#    --num_fewshot 0 \
#    --device 0 \
#    --tasks record

   # --tasks hellaswag,piqa,arc_easy,arc_challenge,openbookqa,winogrande,boolq,cb,copa,wic,wsc,multirc,rte,record,wnli,qqp,cola,mnli,mnli_mismatched,mrpc,sst,qnli
