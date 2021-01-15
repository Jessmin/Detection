from net.fasterrcnn import FasterRCNN
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from utils.utils import AnchorTargetCreator, ProposalTargetCreator
from loss import _fast_rcnn_loc_loss
import torch.nn.functional as F
from collections import namedtuple
import tqdm
from torch.autograd import Variable

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn, optimizer):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 3
        self.roi_sigma = 1

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = optimizer

    def forward(self, imgs, bboxes, labels, scale):
        n = imgs.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        # 获取真实框和标签
        bbox = bboxes[0]
        label = labels[0]

        # 获取公用特征层
        features = self.faster_rcnn.extractor(imgs)
        # 获取faster_rcnn的建议框参数
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        # 获取建议框的置信度和回归系数
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        # ------------------------------------------ #
        #   建议框网络的loss
        # ------------------------------------------ #
        # 先获取建议框网络应该有的预测结果
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox.cpu().numpy(), anchor, img_size)
        gt_rpn_label = torch.Tensor(gt_rpn_label).long()
        gt_rpn_loc = torch.Tensor(gt_rpn_loc)
        # 计算建议框网络的loss值#
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)

        if rpn_score.is_cuda:
            gt_rpn_label = gt_rpn_label.cuda()
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

        # ------------------------------------------ #
        #   classifier网络的loss
        # ------------------------------------------ #
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            bbox.cpu().numpy(),
            label.cpu().numpy(),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)

        if roi_cls_loc.is_cuda:
            roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(), torch.Tensor(gt_roi_label).long()].cuda()
        else:
            roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(), torch.Tensor(gt_roi_label).long()]

        gt_roi_label = torch.Tensor(gt_roi_label).long()
        gt_roi_loc = torch.Tensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        if roi_score.is_cuda:
            gt_roi_label = gt_roi_label.cuda()

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]
        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses


def fit_ont_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    val_toal_loss = 0
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                    boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
                    labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]
                else:
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                    boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)) for box in boxes]
                    labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)) for label in labels]
                train_util.optimizer.zero_grad()
                losses = train_util.forward(imgs, boxes, labels, 1)
                _, _, _, _, val_total = losses
                val_toal_loss += val_total
            pbar.set_postfix(**{'total_loss': val_toal_loss.item() / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))


if __name__ == "__main__":
    # 参数初始化
    annotation_path = '2007_train.txt'
    NUM_CLASSES = 20
    IMAGE_SHAPE = [600, 600, 3]
    BACKBONE = "resnet50"
    model = FasterRCNN(NUM_CLASSES, backbone=BACKBONE)
    # -------------------------------#
    #   Dataloder的使用
    # -------------------------------#
    Use_Data_Loader = True
    Cuda = True
    model_path = r'model_data/voc_weights_resnet.pth'
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('Finished!')

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    if True:
        lr = 1e-4
        Init_Epoch = 0
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        if Use_Data_Loader:
            train_dataset = FRCNNDataset(lines[:num_train], (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
            val_dataset = FRCNNDataset(lines[num_train:], (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
            gen = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=frcnn_dataset_collate)
        # else:
        #     gen = Generator(lines[:num_train], (IMAGE_SHAPE[0], IMAGE_SHAPE[1])).generate()
        #     gen_val = Generator(lines[num_train:], (IMAGE_SHAPE[0], IMAGE_SHAPE[1])).generate()
        epoch_size = num_train
        epoch_size_val = num_val
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.extractor.parameters():
            param.requires_grad = True
        # ------------------------------------#
        #   由于batch==1所以冻结bn层
        # ------------------------------------#
        model.freeze_bn()
        train_util = FasterRCNNTrainer(model, optimizer)

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_ont_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()
