import os
import torchvision
from PIL import Image

from typing import Any, Callable, Optional, Tuple, List, Union

class ArgoverseDetection(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transforms, target_transform, transform)
        db = self.coco.dataset
        self.seqs = db['sequences']
        self.seq_dirs = db['seq_dirs']
        self.coco_mapping = db['coco_mapping']
        self.class_mapping = {
            v: i
            for i, v in enumerate(self.coco_mapping) if v < 80
        }# {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5, 9: 6, 11: 7}
        
        # self.CLASSES = tuple([c['name'] for c in db['categories']])
        # self.cat_ids = self.class_mapping[self.coco.getCatIds(catNms=self.CLASSES)]
        
        img_ids = self.coco.getImgIds()
        self.data_list = {}
        for img_id in img_ids:
            raw_img_info = self.coco.loadImgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            parsed_data_info = self._parse_data_info({
                'raw_img_info':
                raw_img_info
            })
            self.data_list[img_id] = parsed_data_info
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _load_image(self, id: int) -> Image.Image:
        path = self.data_list[id]['img_path']
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    
    def _load_target(self, id: int) -> Image.Image:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def _parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        sid = img_info['sid']
        img_path = os.path.join(self.seq_dirs[sid], img_info['name'])

        # if self.data_prefix.get('seg', None):
        #     seg_map_path = os.path.join(
        #         self.data_prefix['seg'],
        #         img_info['name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        # else:
        #     seg_map_path = None
        data_info['img_path'] = img_path
        data_info['name'] = img_info['name']
        data_info['img_id'] = img_info['img_id']
        # data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        return data_info