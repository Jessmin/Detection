import torch
import numpy as np
import torch.nn.functional as F

def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def nms(detections_class, nms_thres=0.7):
    max_detections = []
    while np.shape(detections_class)[0]:
        # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        max_detections.append(np.expand_dims(detections_class[0], 0))
        if len(detections_class) == 1:
            break
        ious = bbox_iou(max_detections[-1][:, :4], detections_class[1:, :4])[0]
        detections_class = detections_class[1:][ious < nms_thres]
    if len(max_detections) == 0:
        return []
    max_detections = np.concatenate(max_detections, axis=0)
    return max_detections


class DecodeBox():
    def __init__(self, std, mean, num_classes):
        self.std = std
        self.mean = mean
        self.num_classes = num_classes + 1

    def forward(self, roi_cls_locs, roi_scores, rois, height, width, nms_iou, score_thresh):

        rois = torch.Tensor(rois)

        roi_cls_loc = (roi_cls_locs * self.std + self.mean)
        roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])
        roi = rois.view((-1, 1, 4)).expand_as(roi_cls_loc)

        cls_bbox = loc2bbox((roi.cpu().detach().numpy()).reshape((-1, 4)),
                            (roi_cls_loc.cpu().detach().numpy()).reshape((-1, 4)))
        cls_bbox = torch.Tensor(cls_bbox)
        cls_bbox = cls_bbox.view([-1, (self.num_classes), 4])

        # clip bounding box
        cls_bbox[..., 0] = (cls_bbox[..., 0]).clamp(min=0, max=width)
        cls_bbox[..., 2] = (cls_bbox[..., 2]).clamp(min=0, max=width)
        cls_bbox[..., 1] = (cls_bbox[..., 1]).clamp(min=0, max=height)
        cls_bbox[..., 3] = (cls_bbox[..., 3]).clamp(min=0, max=height)

        prob = F.softmax(torch.tensor(roi_scores), dim=1)

        raw_cls_bbox = cls_bbox.cpu().numpy()
        raw_prob = prob.cpu().numpy()

        outputs = []
        arg_prob = np.argmax(raw_prob, axis=1)
        for l in range(1, self.num_classes):
            arg_mask = (arg_prob == l)
            cls_bbox_l = raw_cls_bbox[arg_mask, l, :]
            prob_l = raw_prob[arg_mask, l]

            mask = prob_l > score_thresh

            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]

            if len(prob_l) == 0:
                continue

            label = np.ones_like(prob_l) * (l - 1)
            detections_class = np.concatenate(
                [cls_bbox_l, np.expand_dims(prob_l, axis=-1), np.expand_dims(label, axis=-1)], axis=-1)

            prob_l_index = np.argsort(prob_l)[::-1]
            detections_class = detections_class[prob_l_index]
            nms_out = nms(detections_class, nms_iou)
            if outputs == []:
                outputs = nms_out
            else:
                outputs = np.concatenate([outputs, nms_out], axis=0)
        return outputs
