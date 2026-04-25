_base_ = [
    '../_base_/datasets/hands23_detection.py',
    '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=1)

# ====== WEIGHTS / PRETRAINING ======
# Use Swin-L Co-DINO LVIS checkpoint
load_from = '/lus/lfs1aip2/home/s5a/ahmad.s5a/projects/Co-DETR/checkpoints/co_dino_5scale_lsj_swin_large_3x_lvis.pth'
pretrained = None  # for the Swin backbone itself

# ====== GLOBAL MODEL SETTINGS ======
num_dec_layer = 6
lambda_2 = 2.0
data_root = '/home/s5a/ahmad.s5a/projects/hands23_data/'

# ====== MODEL DEFINITION ======
model = dict(
    type='CoDETR',

    # ----------- BACKBONE: SWIN-L -----------
    backbone=dict(
        type='SwinTransformerV1',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        pretrained=pretrained),

    # ----------- NECK: CHANNEL MAPPER FOR SWIN -----------
    # Map [192, 384, 768, 1536] → 5 levels of 256-dim features
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 192 * 2, 192 * 4, 192 * 8],  # [192, 384, 768, 1536]
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),

    # ----------- AUX RPN HEAD -----------
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0 * num_dec_layer * lambda_2),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=1.0 * num_dec_layer * lambda_2)
    ),

    # ----------- QUERY HEAD (Co-DINO + interaction) -----------
    query_head=dict(
        type='CoDINOHeadWithInteraction',
        num_query=1500,
        num_classes=3,
        num_feature_levels=5,

        # Swin-L last stage channels = 1536
        in_channels=192 * 8,  # 1536

        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        mixed_selection=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4),  # 0.5, 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=300)),
        transformer=dict(
            type='CoDinoTransformer',
            with_pos_coord=True,
            with_coord_feat=False,
            num_co_heads=2,
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                with_cp=6,  # number of layers that use checkpoint
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn', 'norm',
                        'cross_attn', 'norm',
                        'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),

        # main detection losses
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=5.0),
        loss_iou=dict(
            type='GIoULoss',
            loss_weight=2.0),

        # --- INTERACTION LOSS (as you had it) ---
        loss_interaction=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=3,  # you were using 3; adjust if you want 18 etc.
        )
    ),

    # ----------- AUX ROI HEAD (Faster-RCNN style) -----------
    roi_head=[dict(
        type='CoStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=56),
        bbox_head=dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=True,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0 * num_dec_layer * lambda_2),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=10.0 * num_dec_layer * lambda_2)))
    ],

    # ----------- AUX ATSS HEAD -----------
    bbox_head=[dict(
        type='CoATSSHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=1,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0 * num_dec_layer * lambda_2),
        loss_bbox=dict(
            type='GIoULoss',
            loss_weight=2.0 * num_dec_layer * lambda_2),
        loss_centerness=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0 * num_dec_layer * lambda_2)),
    ],

    # ----------- TRAIN / TEST CFG (unchanged) -----------
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
    ],
    test_cfg=[
        dict(
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=8000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.9),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                mask_thr_binary=0.5,
                nms=dict(type='soft_nms', iou_threshold=0.5),
                max_per_img=1000)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='soft_nms', iou_threshold=0.6),
            max_per_img=100),
    ]
)

# ====== DATA / PIPELINES (same as your ViT hands config) ======
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadHandAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            # Policy 1: multiscale resize around 640
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 640), (512, 640), (544, 640),
                               (576, 640), (608, 640), (640, 640),
                               (672, 640), (704, 640), (736, 640)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            # Policy 2: random crop + resize back
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 640), (640, 640), (800, 640)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 640), (512, 640), (544, 640),
                               (576, 640), (608, 640), (640, 640)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleWithHand'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_handside', 'gt_interaction'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    collate_fn=dict(type='custom_collate'),
    train=dict(
        filter_empty_gt=False,
        pipeline=train_pipeline,
        ann_file=data_root + 'annotations/train_h_first_second_full_corrected_auto_80_100.json'),
    val=dict(
        pipeline=test_pipeline,
        ann_file=data_root + 'annotations/val_h_first_second_full_corrected.json'),
    test=dict(
        pipeline=test_pipeline,
        ann_file=data_root + 'annotations/val_h_first_second_full_corrected.json')
)

evaluation = dict(metric='bbox')

# ====== OPTIMIZER / SCHEDULE (same as your ViT config) ======
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[4])

runner = dict(type='EpochBasedRunner', max_epochs=5)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=24,  # still fine to keep 24 here, affects LR decay grouping
        layer_decay_rate=0.8))

custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        momentum=0.0001,
        priority=49),
]
