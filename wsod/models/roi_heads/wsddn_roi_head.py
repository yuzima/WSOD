from typing import List, Tuple
import torch
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2result, bbox2roi
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList
from mmdet.models import BaseRoIHead
from mmdet.models.roi_heads import BBoxTestMixin, MaskTestMixin
from mmdet.models.utils import unpack_gt_instances


@MODELS.register_module()
class WSDDNRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):

    def init_assigner_sampler(self):
        pass

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
                       bbox_head: ConfigType) -> None:
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        pass

    def init_weights(self):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.bbox_roi_extractor.init_weights()
        self.bbox_head.init_weights()

    def forward(self, 
                      x: Tuple[Tensor],
                      rpn_results_list: InstanceList,):
        """Dummy forward function."""
        # bbox head
        results = ()
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            results = results + (bbox_results['cls_score'],
                                 bbox_results['bbox_pred'])
        return results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList):
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        
        gt_labels = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        losses = dict()

        # bbox head forward and loss
        bbox_results = self.bbox_loss(
            x, rpn_results_list, gt_labels, batch_img_metas)
        losses.update(bbox_results['loss_bbox'])
        return losses

    def bbox_loss(self, x, rpn_results_list, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in rpn_results_list])
        bbox_results = self._bbox_forward(x, rois)

        loss_bbox = self.bbox_head.loss(
            bbox_results['cls1'], bbox_results['cls2'], gt_labels)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        cls1, cls2 = self.bbox_head(bbox_feats)

        bbox_results = dict(cls1=cls1, cls2=cls2)
        return bbox_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list,
                     rcnn_test_cfg,
                     rescale=False) -> InstanceList:
        """Test only det bboxes without augmentation."""
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in batch_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in batch_img_metas)

        # split batch bbox prediction back to each image
        cls1 = bbox_results['cls1']
        cls2 = bbox_results['cls2']
        num_proposals_per_img = tuple(len(p) for p in proposals)

        rois = rois.split(num_proposals_per_img, 0)
        cls1 = cls1.split(num_proposals_per_img, 0)
        cls2 = cls2.split(num_proposals_per_img, 0)

        bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        result_list = []
        for i in range(len(proposals)):
            result = self.bbox_head.predict_by_feat(
                rois[i],
                cls1[i],
                cls2[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            result_list.append(result)
        return result_list

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        assert self.with_bbox, 'Bbox head must be implemented.'

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        result_list = self.predict_bbox(
            x, batch_img_metas, rpn_results_list, self.test_cfg, rescale=rescale)

        return result_list

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        return [bbox_results]
