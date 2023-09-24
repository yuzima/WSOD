import copy
import os.path as osp

import numpy as np
from .api_wrappers import COCO
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset
from mmcv.transforms import Compose
from mmengine.fileio import get_local_path
from typing import List, Union


@DATASETS.register_module()
class AVHDDataset(CocoDataset):

    COCOAPI = COCO

    # def __init__(self,
    #              ann_file,
    #              pipeline,
    #              classes=None,
    #              data_root=None,
    #              img_prefix='',
    #              seg_prefix=None,
    #              proposal_file=None,
    #              test_mode=False,
    #              filter_cfg=None,
    #              take=None,
    #              skip=None,
    #              backend_args=None,
    #              indices=None):
    #     self.ann_file = ann_file
    #     self.data_root = data_root
    #     self.img_prefix = img_prefix
    #     self.seg_prefix = seg_prefix
    #     self.proposal_file = proposal_file
    #     self.test_mode = test_mode
    #     self.filter_cfg = filter_cfg
    #     self.CLASSES = classes
    #     self.take = take
    #     self.skip = skip
    #     self.backend_args = backend_args
    #     self._indices = indices

    #     # join paths if data_root is specified
    #     if self.data_root is not None:
    #         if not osp.isabs(self.ann_file):
    #             self.ann_file = osp.join(self.data_root, self.ann_file)
    #         if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
    #             self.img_prefix = osp.join(self.data_root, self.img_prefix)
    #         if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
    #             self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
    #         if not (self.proposal_file is None
    #                 or osp.isabs(self.proposal_file)):
    #             self.proposal_file = osp.join(self.data_root,
    #                                           self.proposal_file)
    #     # load annotations (and proposals)
    #     # self.data_list = self.load_data_list()

    #     # if self.proposal_file is not None:
    #     #     self.proposals = self.load_proposals(self.proposal_file)
    #     # else:
    #     #     self.proposals = None

    #     # # filter images too small and containing no annotations
    #     # if not test_mode:
    #     #     self.data_list = self.filter_data()
    #     #     # self.data_infos = [self.data_infos[i] for i in valid_inds]
    #     #     # if self.proposals is not None:
    #     #     #     self.proposals = [self.proposals[i] for i in valid_inds]
    #     #     # set group flag for the sampler
    #     #     self._set_group_flag()

    #     # processing pipeline
    #     self.pipeline = Compose(pipeline)

    def load_data_list(self):
        # self.coco = COCO(ann_file)
        # db = self.coco.dataset

        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
            db = self.coco.dataset

        self.CLASSES = tuple([c['name'] for c in db['categories']])
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes']) # [0, 1, 2, 3, 4, 5, 6, 7]
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        # print(self.coco.cat_img_map)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        # self.coco_mapping = db.get('coco_mapping', None)
        # self.class_mapping = {
        #     i: v
        #     for i, v in enumerate(db.get('coco_mapping', None)) if v < 80
        # } # {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5, 9: 6, 11: 7}

        self.seqs = db['sequences']
        self.seq_dirs = db['seq_dirs']
        # self.img_ids = self.coco.getImgIds()
        img_ids = self.coco.get_img_ids()

        data_list = []
        # for img in self.coco.imgs.values():
        for img_id in img_ids:
            # img_name = img['name']
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })

            # sid = img['sid']
            # img_path = osp.join(self.data_prefix['img'], self.seq_dirs[sid], img['name'])
            # img['img_path'] = img_path
            # img['img_id'] = img['id']
            data_list.append(parsed_data_info)

        # if self.skip is not None:
        #     self.img_ids = self.img_ids[::self.skip]
        #     data_list = data_list[::self.skip]
        # if self.take is not None:
        #     self.img_ids = self.img_ids[:self.take]
        #     data_list = data_list[:self.take]
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        sid = img_info['sid']
        # img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        img_path = osp.join(self.data_prefix['img'], self.seq_dirs[sid], img_info['name'])

        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1


@DATASETS.register_module()
class AVHDGTDataset(AVHDDataset):
    """AVHDDataset, but returns images with ground
    truth annotations during testing."""

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_data(idx)
        while True:
            data = self.prepare_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data


@DATASETS.register_module()
class AVHDPrevDetDataset(AVHDDataset):
    """AVHDDataset, but with helpers to facilitate online testing."""

    def result_to_det(self, result):
        if isinstance(result, tuple):
            # contains mask
            result = result[0]

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels).astype(np.int64)
        n_det = len(labels)
        if n_det:
            bboxes = np.vstack(result)
            scores = bboxes[:, -1].astype(np.float32)
            bboxes = bboxes[:, :4].astype(np.float32)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

        return dict(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
            bboxes_ignore=None,
            masks=n_det*[None],
            seg_map=n_det*[None],
        )

    def prepare_test_img_with_prevdets(self, idx, prevdets):
        """
        Returns image at index 'idx' with:
        - empty annotations if this image is the first frame of its sequence
        - prevdets annotations else
        """
        img_info = self.data_infos[idx]
        pidx = self.get_prev_idx(idx)
        if pidx is None:
            det = dict(
                bboxes=np.zeros((0, 4), dtype=np.float32),
                labels=np.array([], dtype=np.int64),
                scores=np.array([], dtype=np.float32),
                bboxes_ignore=None,
                masks=[],
                seg_map=[],
            )
        else:
            det = prevdets
        results = dict(img_info=img_info, ann_info=det)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_prev_idx(self, idx):
        img = self.data_infos[idx]
        if img['fid'] == 0:
            return None
        else:
            return idx - 1
