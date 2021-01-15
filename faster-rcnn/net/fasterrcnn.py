import torch.nn as nn
import torch
from resnet50 import get_resnet
from rpn import RegionProposalNetwork
from classifier import Resnet50RoIHead


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, mode='training', loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 feat_stride=16, anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2], backbone='resnet50'):
        super(FasterRCNN, self).__init__()
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.feat_stride = feat_stride
        self.extractor, classifier = get_resnet()
        self.rpn = RegionProposalNetwork(
            1024, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            mode=mode
        )
        self.head = Resnet50RoIHead(
            n_class=num_classes + 1,
            roi_size=14,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head.forward(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
