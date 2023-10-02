import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.models.layers import multiclass_nms
from mmdet.registry import MODELS
from mmengine.structures import InstanceData
from mmdet.structures.bbox import scale_boxes

@MODELS.register_module()
class WSDDNHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 hidden_channels=1024,
                 num_classes=20):
        super(WSDDNHead, self).__init__()
        self.roi_feat_size = _pair(roi_feat_size)
        # 7 * 7 = 49
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fp16_enabled = False

        in_channels *= self.roi_feat_area  # 512 * 49 = 25088

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout2 = nn.Dropout()

        self.fc_cls1 = nn.Linear(hidden_channels, num_classes)
        self.fc_cls2 = nn.Linear(hidden_channels, num_classes)

        self.eps = 1e-5
        # self.weakly_multiclass_nms = WeaklyMulticlassNMS(20)

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc_cls1.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls1.bias, 0)
        nn.init.normal_(self.fc_cls2.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls2.bias, 0)

    # @auto_fp16()
    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls1 (Tensor): Classification scores.
                - cls2 (Tensor): Detection scores.
        """

        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        cls1 = self.fc_cls1(x)
        cls2 = self.fc_cls2(x)
        return cls1, cls2

    # @force_fp32(apply_to=('cls1', 'cls2'))
    def loss(self,
             cls1,
             cls2,
             labels):
        losses = dict()

        cls1 = F.softmax(cls1, dim=1)
        cls2 = F.softmax(cls2, dim=0)

        cls = cls1 * cls2
        cls = cls.sum(dim=0)
        cls = torch.clamp(cls, 0., 1.)

        num_classes = cls.size(0)
        labels = torch.cat(labels, dim=0)

        probability = torch.zeros(num_classes, device=labels.device)
        for i, label in enumerate(labels):
            probability[label.to(torch.int64)] = 1

        labels = probability

        if num_classes >= labels.size(0):
            labels = torch.cat((labels, torch.zeros(
                num_classes - labels.size(0), device=labels.device)), dim=0)
        if num_classes < labels.size(0):
            labels = torch.narrow(labels, 0, 0, num_classes)

        loss_wsddn = F.binary_cross_entropy(cls, labels.float(), reduction='sum')

        losses = dict()
        losses['loss_wsddn'] = loss_wsddn
        return losses

    # @force_fp32(apply_to=('cls1', 'cls2'))
    def predict_by_feat(self,
                        rois,
                        cls1,
                        cls2,
                        bbox_pred,
                        img_meta, 
                        rescale=False,
                        rcnn_test_cfg=None) -> InstanceData:

        cls1 = F.softmax(cls1, dim=1)
        cls2 = F.softmax(cls2, dim=0)

        scores = cls1 * cls2
        scores_pad = torch.zeros(
            (scores.shape[0], 1), dtype=torch.float32).to(device=scores.device)
        scores = torch.cat([scores, scores_pad], dim=1)
        img_shape = img_meta['img_shape']

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            # assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        results = InstanceData()
        if rcnn_test_cfg is None:
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes, scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results
