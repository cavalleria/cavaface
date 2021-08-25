#!/usr/bin/env bash
export PYTHONDONTWRITEBYTECODE=1

MODEL="/workspace/results/lyb_results/IR_SE_100_Arcface/model_ep24_1.pth,24"
python -u ./main.py \
    --eval_sets="megaface" \
    --model_type=pytorch_fp32 \
    --gpus "1,2,3" \
    --net_scale "large" \
    --model_path=${MODEL}
echo "Eval model: ${MODEL}, done!"
MODEL="/workspace/results/lyb_results/IR_SE_100_Arcface/model_ep24_1.pth,24"
python -u ./main.py \
    --eval_sets="ijbc" \
    --model_type=pytorch_fp32 \
    --gpus "1,2,3" \
    --net_scale "large" \
    --model_path=${MODEL}
echo "Eval model: ${MODEL}, done!"
