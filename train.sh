#!/usr/bin/env bash
img=/workspace/dataset/AIM-Train/image
tmp=/workspace/dataset/AIM-Train/trimap
seg_img=/workspace/dataset/DUTS/DUTS-TR/DUTS-TR/DUTS-TR-Image
seg_mask=/workspace/dataset/DUTS/DUTS-TR/DUTS-TR/DUTS-TR-Trimap
val_img=/workspace/dataset/AIM-500/original_png
val_tmp=/workspace/dataset/AIM-500/trimap
val_out=data/val
ckpt=checkpoints
patch_size=320
sample=100
epoch=100

batch=2

t=

model=g

lr=1e-4

gpu=0

CUDA_VISIBLE_DEVICES=${gpu} python train.py -dgr -m=t-net --lr=${lr} --model=${model} --batch=${batch} --tolerance_loss=${t} --img=${img} --trimap=${tmp} --seg_img=${seg_img} --seg_mask=${seg_mask} --val-out=${val_out} --val-img=${val_img} --val-trimap=${val_tmp} --ckpt=${ckpt} --patch-size=${patch_size} --sample=${sample} --epoch=${epoch}
