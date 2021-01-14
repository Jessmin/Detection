import torch.nn as nn
import torch
from resnet50 import get_resnet

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

        pass
