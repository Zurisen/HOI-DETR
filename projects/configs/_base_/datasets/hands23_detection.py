# dataset settings
dataset_type = 'Hands23CocoDataset'   # <--- CHANGED

data_root = 'data/hands23_data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

custom_imports = dict(
    imports=['mmdet.datasets.hands23_coco'],
    allow_failed_imports=False
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadHandAnnotations', with_bbox=True),  # <-- custom
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleWithHand'),  # <-- custom bundle
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_handside','gt_bboxes_all' ,'gt_interaction']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_h_first_second.json',
        img_prefix=data_root + 'allMergedBlur/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_h_first_second.json',
        img_prefix=data_root + 'allMergedBlur/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_h_first_second.json',
        img_prefix=data_root + 'allMergedBlur/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
