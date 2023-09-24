dataset_type = 'AVHDDataset'
data_root = 'data/Argoverse/'

original_image_shape = (1200, 1920)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Timer', name='preprocessing', transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
            ]),
            dict(
                type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
                meta_keys=[
                    'filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg', 'preprocessing_time']),
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Argoverse-HD/annotations/train.json',
        img_prefix='Argoverse-1.1/tracking',
        pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Argoverse-HD/annotations/val.json',
        img_prefix='Argoverse-1.1/tracking',
        pipeline=test_pipeline)
)
test_dataloader = val_dataloader

val_evaluator = dict(metric=['bbox'], classwise=True)
test_evaluator = val_evaluator