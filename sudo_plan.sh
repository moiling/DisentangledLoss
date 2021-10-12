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
epoch=200
batch=3
t=
model=g
sample1=2000
sample2=1000
sample3=500
lr1=1e-6
lr2=5e-7
lr3=1e-7
cur_path=$(pwd)
validate_path=../base/
validate_file=validate.py
name=disentangled_base_normal_best
gpu=0

mkdir checkpoints_old
mkdir data_old
#-------------------------------------------------------------------------------------
# train: sample=2000, lr=1e-6
CUDA_VISIBLE_DEVICES=${gpu} python train.py -dgr -m=t-net --lr=${lr1} --model=${model} --batch=${batch} --tolerance_loss=${t} --img=${img} --trimap=${tmp} --seg_img=${seg_img} --seg_mask=${seg_mask} --val-out=${val_out} --val-img=${val_img} --val-trimap=${val_tmp} --ckpt=${ckpt} --patch-size=${patch_size} --sample=${sample1} --epoch=${epoch}
# save ckpt to local.
# sudo (docker needn't sudo)
cp checkpoints/t-net-best.pt checkpoints_old/${name}-s-${sample1}-lr-${lr1}.pt
# validate:
cd ${validate_path} || exit
CUDA_VISIBLE_DEVICES=${gpu} python ${validate_file} -dg -m=t-net --model=${model} --tolerance_loss=${t} --val_img=${val_img} --val_trimap=${val_tmp} --val_matte=${val_mat} --val_fg=${val_fg} --val_bg=${val_bg} --out=${val_out} --ckpt_path=${cur_path}/${ckpt}/t-net-best.pt --patch-size=10000
cd ${cur_path} || exit
# save data to local.
# sudo
cp -r data/val/trimap/ data_old/${name}-s-${sample1}-lr-${lr1}/
# save date to /share. (docker can't visit /share)
# sudo cp -r data/val/trimap/ /share/data/${name}-s-${sample1}-lr-${lr1}/
# sudo
mv checkpoints/t-net-best.pt checkpoints/${name}-s-${sample1}-lr-${lr1}.pt

#-------------------------------------------------------------------------------------
# train: sample=1000, lr=5e-7
CUDA_VISIBLE_DEVICES=${gpu} python train.py -dgr -m=t-net --lr=${lr2} --model=${model} --batch=${batch} --tolerance_loss=${t} --img=${img} --trimap=${tmp} --seg_img=${seg_img} --seg_mask=${seg_mask} --val-out=${val_out} --val-img=${val_img} --val-trimap=${val_tmp} --ckpt=${ckpt} --patch-size=${patch_size} --sample=${sample2} --epoch=${epoch}
# save ckpt to local.
# sudo
cp checkpoints/t-net-best.pt checkpoints_old/${name}-s-${sample2}-lr-${lr2}.pt
# validate:
cd ${validate_path} || exit
CUDA_VISIBLE_DEVICES=${gpu} python v${validate_file} -dg -m=t-net --model=${model} --tolerance_loss=${t} --val_img=${val_img} --val_trimap=${val_tmp} --val_matte=${val_mat} --val_fg=${val_fg} --val_bg=${val_bg} --out=${val_out} --ckpt_path=${cur_path}/${ckpt}/t-net-best.pt --patch-size=10000
cd ${cur_path} || exit
# save data to local.
# sudo
cp -r data/val/trimap/ data_old/${name}-s-${sample2}-lr-${lr2}/
# save date to /share.
# sudo cp -r data/val/trimap/ /share/data/${name}-s-${sample2}-lr-${lr2}/
# sudo
mv checkpoints/t-net-best.pt checkpoints/${name}-s-${sample2}-lr-${lr2}.pt

#-------------------------------------------------------------------------------------
# train: sample=500, lr=1e-7
CUDA_VISIBLE_DEVICES=${gpu} python train.py -dgr -m=t-net --lr=${lr3} --model=${model} --batch=${batch} --tolerance_loss=${t} --img=${img} --trimap=${tmp} --seg_img=${seg_img} --seg_mask=${seg_mask} --val-out=${val_out} --val-img=${val_img} --val-trimap=${val_tmp} --ckpt=${ckpt} --patch-size=${patch_size} --sample=${sample3} --epoch=${epoch}
# save ckpt to local.
# sudo
cp checkpoints/t-net-best.pt checkpoints_old/${name}-s-${sample3}-lr-${lr3}.pt
# validate:
cd ${validate_path} || exit
CUDA_VISIBLE_DEVICES=${gpu} python ${validate_file} -dg -m=t-net --model=${model} --tolerance_loss=${t} --val_img=${val_img} --val_trimap=${val_tmp} --val_matte=${val_mat} --val_fg=${val_fg} --val_bg=${val_bg} --out=${val_out} --ckpt_path=${cur_path}/${ckpt}/t-net-best.pt --patch-size=10000
cd ${cur_path} || exit
# save data to local.
# sudo
cp -r data/val/trimap/ data_old/${name}-s-${sample3}-lr-${lr3}/
# save date to /share.
# sudo cp -r data/val/trimap/ /share/data/${name}-s-${sample3}-lr-${lr3}/
# sudo
mv checkpoints/t-net-best.pt checkpoints/${name}-s-${sample2}-lr-${lr2}.pt