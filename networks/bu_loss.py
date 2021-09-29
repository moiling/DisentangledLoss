import torch
import torch.nn as nn
import torch.nn.functional as F


def bu_loss(pred_trimap_prob, gt_bu_2, tolerance_loss=False):
    loss = bu_class_loss(pred_trimap_prob, gt_bu_2)

    if tolerance_loss:
        loss += 10 * tolerance_bu_loss(pred_trimap_prob, gt_bu_2)
    return loss


def bu_class_loss(pred_trimap_prob, gt_bu_2):
    gt_bu_type = gt_bu_2.argmax(dim=1)
    # trimap => BUF => U = U +F

    softmax_func = nn.Softmax(dim=1)
    pred_trimap_softmax = softmax_func(pred_trimap_prob)

    b = pred_trimap_softmax[:, 0:1, ...]
    u = pred_trimap_softmax[:, 1:2, ...]
    f = pred_trimap_softmax[:, 2:3, ...]

    u = u + f

    pred_softmax = torch.cat((b, u), dim=1)

    nll_loss_func = nn.NLLLoss()
    cross_entropy_loss = nll_loss_func(torch.log(pred_softmax), gt_bu_type)

    return cross_entropy_loss


def tolerance_bu_loss(pred_trimap_prob, gt_bu_2):
    gt_type = torch.zeros_like(gt_bu_2.argmax(dim=1))  # [B, H, W], all target type = 0
    # CrossEntropyLoss = softmax + log + NLLLoss
    # soft max
    softmax_func = nn.Softmax(dim=1)
    pred_trimap_softmax = softmax_func(pred_trimap_prob)
    # BUF
    b = pred_trimap_softmax[:, 0:1, ...]
    u = pred_trimap_softmax[:, 1:2, ...]
    f = pred_trimap_softmax[:, 2:3, ...]

    gt_argmax = gt_bu_2.argmax(dim=1).unsqueeze(dim=1)
    b_mask = (gt_argmax == 0)
    u_mask = (gt_argmax == 1)

    # B => true = b+u, false = f
    # U => true = u+f, false = b
    pred_true_softmax = b_mask * (b + u) + u_mask * (u + f)
    pred_false_softmax = b_mask * f + u_mask * b

    pred_softmax = torch.cat((pred_true_softmax, pred_false_softmax), dim=1)  # [B, 2, H, W]

    # log + NLLLoss
    nll_loss_func = nn.NLLLoss()
    cross_entropy_loss = nll_loss_func(torch.log(pred_softmax), gt_type)
    return cross_entropy_loss
