@echo off

::set img=D:/Dataset/ZT/zt4k/train/image
::set tmp=D:/Dataset/ZT/zt4k/train/trimap
::set mat=D:/Dataset/ZT/zt4k/train/alpha
::set fg=D:/Dataset/ZT/zt4k/train/fg
::set bg=D:/Dataset/ZT/zt4k/train/bg
::set val_img=D:/Dataset/ZT/zt4k/test/image
::set val_tmp=D:/Dataset/ZT/zt4k/test/trimap
::set val_mat=D:/Dataset/ZT/zt4k/test/alpha
::set val_fg=D:/Dataset/ZT/zt4k/test/fg
::set val_bg=D:/Dataset/ZT/zt4k/test/bg
set img=D:/Dataset/Matting/AIM-Train/image
set tmp=D:/Dataset/Matting/AIM-Train/trimap
set seg_img=D:/Dataset/SOC/DUTS/DUTS-TR/DUTS-TR/DUTS-TR-Image
set seg_mask=D:/Dataset/SOC/DUTS/DUTS-TR/DUTS-TR/DUTS-TR-Trimap
set val_img=D:/Dataset/Matting/AIM-500/original_png
set val_tmp=D:/Dataset/Matting/AIM-500/trimap
set val_out=data/val
set ckpt=checkpoints
set patch_size=320
set sample=100
set epoch=100

set batch=2

set t=

set model=a

set lr=1e-6

python train.py -dgr -m=t-net --lr=%lr% --model=%model% --batch=%batch%  --tolerance_loss=%t% --img=%img% --trimap=%tmp% --seg_img=%seg_img% --seg_mask=%seg_mask% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%
