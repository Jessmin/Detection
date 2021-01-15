import torch


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape)
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1

    if pred_loc.is_cuda:
        gt_loc = gt_loc.cuda()
        in_weight = in_weight.cuda()
    # smooth_l1损失函数
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # 进行标准化
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()
