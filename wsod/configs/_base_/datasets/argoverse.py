# dataset settings
dataset_type = 'AVHDGTDataset'
data_root = 'data/Argoverse/'
classes = ('person', 'bicycle', 'car', 'motorcycle',
           'bus', 'truck', 'traffic_light', 'stop_sign')

original_image_shape = (1200, 1920)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'pad_shape', 'flip', 'flip_direction', 'img_norm_cfg',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='Argoverse-HD/annotations/train_new.json',
        data_prefix=dict(img='Argoverse-1.1/tracking'),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='Argoverse-HD/annotations/val_new.json',
        data_prefix=dict(img='Argoverse-1.1/tracking'),
        pipeline=test_pipeline,
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='Argoverse-HD/annotations/test.json',
        data_prefix=dict(img='Argoverse-1.1/tracking'),
        pipeline=test_pipeline,
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'Argoverse-HD/annotations/val_new.json',
    metric='bbox',
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'Argoverse-HD/annotations/test.json',
    metric='bbox',
    format_only=False)

evaluation = dict(interval=1, metric=['mAP', 'bbox'], classwise=True)