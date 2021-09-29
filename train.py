from tensorboardX import SummaryWriter

import torch
import random
import logging
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

import utils
from dataloader.dataset import TrainDataset, ValDataset
from dataloader.prefetcher import PreFetcher
from networks.loss import trimap_loss
from networks.matting_model import MattingModel

"""================================================== Arguments ================================================="""
parser = argparse.ArgumentParser('Portrait Matting Training Arguments.')

parser.add_argument('--img',        type=str,   default='', help='training images.')
parser.add_argument('--trimap',     type=str,   default='', help='intermediate trimaps.')
parser.add_argument('--seg_img',    type=str,   default='', help='seg images.')
parser.add_argument('--seg_mask',   type=str,   default='', help='seg masks')
parser.add_argument('--val-out',    type=str,   default='',   help='val image out.')
parser.add_argument('--val-img',    type=str,   default='',   help='val images.')
parser.add_argument('--val-trimap', type=str,   default='',   help='intermediate val trimaps.')
parser.add_argument('--ckpt',       type=str,   default='',   help='checkpoints.')
parser.add_argument('--batch',      type=int,   default=2,    help='input batch size for train')
parser.add_argument('--val-batch',  type=int,   default=1,    help='input batch size for val')
parser.add_argument('--epoch',      type=int,   default=10,   help='number of epochs.')
parser.add_argument('--sample',     type=int,   default=-1,   help='number of samples. -1 means all samples.')
parser.add_argument('--lr',         type=float, default=1e-5, help='learning rate while training.')
parser.add_argument('--patch-size', type=int,   default=480,  help='patch size of input images.')
parser.add_argument('--seed',       type=int,   default=42,   help='random seed.')

parser.add_argument('--model', type=str, choices=['p', 'm', 'g'], default='p', help='p = PSPNet, m = MobileNetV2, g = GFM')

parser.add_argument('--tolerance_loss',       type=bool,   default=False,   help='tolerance loss.')

parser.add_argument('-d', '--debug',         action='store_true', help='log for debug.')
parser.add_argument('-g', '--gpu',           action='store_true', help='use gpu.')
parser.add_argument('-r', '--resume',        action='store_true', help='load a previous checkpoint if exists.')
parser.add_argument('--hr',                  action='store_true', help='lr or hr.')

parser.add_argument('-m', '--mode', type=str, choices=['end2end', 'f-net', 'm-net', 't-net'], default='t-net', help='working mode.')

args = parser.parse_args()

"""================================================= Presetting ================================================="""
torch.set_flush_denormal(True)  # flush cpu subnormal float.
cudnn.enabled = True
cudnn.benchmark = True
# random seed.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)-15s] [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger('train')
tb_logger = SummaryWriter()
if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logger.debug(args)

"""================================================ Load DataSet ================================================"""
# train
train_data = TrainDataset(args)
train_data_loader = DataLoader(train_data, batch_size=args.batch, drop_last=True, shuffle=True)
train_data_loader = PreFetcher(train_data_loader)
# val
val_data = ValDataset(args)
val_data_loader = DataLoader(val_data, batch_size=args.val_batch)

"""================================================ Build Model ================================================="""
matting_model = MattingModel(args.lr, not (args.gpu and torch.cuda.is_available()), args.mode, args.model)

if args.resume:
    matting_model.resume(args.ckpt)

"""================================================= Main Loop =================================================="""
for epoch in range(matting_model.start_epoch, args.epoch + 1):
    """--------------- Train ----------------"""
    matting_model.train()
    logger.info(f'Epoch: {epoch}/{args.epoch}')

    for idx, batch in enumerate(train_data_loader):
        """ Load Batch Data """
        img         = batch['img']
        trimap_2    = batch['trimap_2']
        seg_img     = batch['seg_img']
        seg_mask_2  = batch['seg_mask_2']

        if args.gpu and torch.cuda.is_available():
            img         = img.cuda()
            trimap_2    = trimap_2.cuda()
            seg_img     = seg_img.cuda()
            seg_mask_2  = seg_mask_2.cuda()

        else:
            img         = img.cpu()
            trimap_2    = trimap_2.cpu()
            seg_img     = seg_img.cpu()
            seg_mask_2  = seg_mask_2.cpu()

        """ Forward """
        pred = matting_model.forward(img)
        seg_pred = matting_model.forward(seg_img)

        """ Backward """
        loss = matting_model.backward(pred, seg_pred, trimap_2, seg_mask_2, tolerance_loss=args.tolerance_loss)

        """ Write Log and Tensorboard """
        logger.debug(f'{args.mode}\t Batch: {idx + 1}/{len(train_data_loader.orig_loader)} \t'
                     f'loss: {loss.item():8.5f} ')

        step = (epoch - 1) * len(train_data_loader.orig_loader) + idx
        if step % 100 == 0:
            tb_logger.add_scalar('TRAIN/Loss', loss.item(), step)

    """------------ Validation --------------"""
    matting_model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_data_loader):
            """ Load Batch Data """
            img         = batch['img']
            trimap_3    = batch['trimap_3']

            if args.gpu and torch.cuda.is_available():
                img         = img.cuda()
                trimap_3    = trimap_3.cuda()

            else:
                img         = img.cpu()
                trimap_3    = trimap_3.cpu()

            """ Forward """
            ptp = matting_model.forward(img)

            """ Calculate Loss """
            loss = trimap_loss(ptp, trimap_3, tolerance_loss=args.tolerance_loss)

            val_loss += loss.item()

            """ Write Log and Save Images """
            logger.debug(f'Batch: {idx + 1}/{len(val_data_loader)} \t' 
                         f'Validation Loss: {loss.item():8.5f} \t')

            utils.save_images(args.val_out, batch['name'], ptp, logger)

    average_loss = val_loss / len(val_data_loader)
    matting_model.losses.append(average_loss)

    """ Write Log and Tensorboard """
    tb_logger.add_scalar('TEST/Loss', average_loss, epoch)
    logger.info(f'Loss:{average_loss:8.5f}')

    """------------ Save Model --------------"""
    if min(matting_model.losses) == average_loss:
        logger.info('Minimal loss so far.')
        matting_model.save(args.ckpt, epoch, best=True)
    else:
        matting_model.save(args.ckpt, epoch, best=False)
