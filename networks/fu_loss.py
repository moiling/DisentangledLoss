import torch
import torch.nn as nn
import torch.nn.functional as F

def fu_loss(pred_trimap_prob, gt_bu_2, tolerance_loss=False):
    loss = fu_class_loss(pred_trimap_prob, gt_bu_2)

    if tolerance_loss:
        loss += 10 * tolerance_fu_loss(pred_trimap_prob, gt_bu_2)
    return loss


def fu_class_loss(pred_trimap_prob, gt_bu_2):
    gt_bu_type = gt_bu_2.argmax(dim=1)
    # GT => UF
    # trimap => BUF => U = U + B

    softmax_func = nn.Softmax(dim=1)
    pred_trimap_softmax = softmax_func(pred_trimap_prob)

    b = pred_trimap_softmax[:, 0:1, ...]
    u = pred_trimap_softmax[:, 1:2, ...]
    f = pred_trimap_softmax[:, 2:3, ...]

    u = u + b

    pred_softmax = torch.cat((u, f), dim=1)

    nll_loss_func = nn.NLLLoss()
    cross_entropy_loss = nll_loss_func(torch.log(pred_softmax), gt_bu_type)

    return cross_entropy_loss


def tolerance_fu_loss(pred_trimap_prob, gt_bu_2):
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
    u_mask = (gt_argmax == 0)
    f_mask = (gt_argmax == 1)

    # U => true = b+u, false = f
    # F => true = u+f, false = b
    pred_true_softmax = u_mask * (b + u) + f_mask * (u + f)
    pred_false_softmax = u_mask * f + f_mask * b

    pred_softmax = torch.cat((pred_true_softmax, pred_false_softmax), dim=1)  # [B, 2, H, W]

    # log + NLLLoss
    nll_loss_func = nn.NLLLoss()
    cross_entropy_loss = nll_loss_func(torch.log(pred_softmax), gt_type)
    return cross_entropy_loss
