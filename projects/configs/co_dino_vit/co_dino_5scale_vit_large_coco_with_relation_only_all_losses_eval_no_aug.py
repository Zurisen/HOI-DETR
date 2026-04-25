_base_ = [
    '../_base_/datasets/hands23_detection.py',
    '../_base_/default_runtime.py'
]
checkpoint_config = dict(interval=2)
#load_from = '/gpfs/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/checkpoints/co-dino-coco-detect.pth'
#load_from = '/gpfs/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_full_02/epoch_3.pth'
#load_from = '/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_00/epoch_1_done.pth'
#load_from ='/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_full_dummy/epoch_7_interaction.pth'
#load_from ='/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_inter_10_percent_ce_1/epoch_7.pth'
#load_from ='/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_inter_10_percent_ce_2_geo_2c_11_first/epoch_7.pth'
#load_from ='/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_inter_10_percent_focal_12_order/epoch_7.pth'
#load_from ='/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_inter_10_percent_ce_1_geometric/epoch_7.pth'
#load_from='/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_inter_10_percent_ce_2_geo_2c_11_first_freez/epoch_4.pth'
#load_from='/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_inter_10_percent_ce_2_geo_2c_11_first_freez_batch_16/epoch_10.pth'
#load_from = '/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_inter_10_percent_ce_2_geo_2c_11_first_complete/epoch_7.pth'
#load_from='/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_inter_10_percent_ce_2_geo_2c_11_first_freez_batch_16/epoch_12.pth'
#load_from='/home/bris/bris456520/ehpc298/Ahmad/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_inter_10_percent_ce_2_geo_2c_11_first2/epoch_7.pth'
#load_from = '/home/s5a/ahmad.s5a/projects/Co-DETR/work_dirs/co_dino_5scale_vit_large_coco_low_inter_10_percent_ce_2_geo_2c_11_first2_fixed/epoch_10.pth'
#load_from='/home/s5a/ahmad.s5a/projects/Co-DETR/work_dirs/inter_10_percent_ce_2_geo_sin/epoch_10.pth'
#load_from='/home/s5a/ahmad.s5a/projects/Co-DETR/work_dirs/inter_10_percent_ce_2_geo_sin_fs/epoch_3.pth'
#load_from='/home/s5a/ahmad.s5a/projects/Co-DETR/work_dirs/inter_10_percent_ce_2_geo_sin_q_only/epoch_10.pth'
#load_from='work_dirs/inter_full_low_2e4_corrected/epoch_2.pth'


#load_from='work_dirs/inter_full_low_1e4_corrected_auto80_100_all_pairs/epoch_5.pth'


load_from='work_dirs/01_inter_full_low_1e4_corrected_auto80_100_h_f_s_w_3/epoch_5.pth'
#load_from='work_dirs/01_inter_full_low_1e4_corrected_auto80_100_h_f_s_w_3_interaction_ony/epoch_5.pth'



#load_from='work_dirs/11_inter_full_low_1e4_corrected_auto80_100_h_f_s_all_pairs_N_2_ordered_w_3/epoch_3.pth'
#load_from='work_dirs/12_inter_full_low_1e4_corrected_auto80_100_h_f_s_w_3_lvis_pretraining/epoch_5.pth'
#load_from='work_dirs/09_inter_full_low_1e4_corrected_auto80_100_h_f_s_all_pairs_N_2/epoch_4.pth'
#load_from='work_dirs/11_inter_full_low_1e4_corrected_auto80_100_h_f_s_all_pairs_N_2_ordered/epoch_4.pth'
#load_from='work_dirs/01_inter_full_low_2e4_corrected_auto80_100_h_f_s_1e4/epoch_5.pth'
#load_from='work_dirs/11_inter_full_low_1e4_corrected_auto80_100_h_f_s_all_pairs_N_2_ordered_w_3/epoch_5.pth'
#load_from='work_dirs/01_inter_full_low_1e4_corrected_auto80_100_h_f_s_w_3/epoch_5.pth'
#load_from='work_dirs/08_inter_full_low_1e4_corrected_auto80_100_h_f_s_old_w_3/epoch_5.pth'
#load_from='work_dirs/07_inter_full_low_1e4_corrected_auto80_100_h_f_s_10_percent/epoch_5.pth'
#load_from='/home/s5a/ahmad.s5a/projects/Co-DETR/work_dirs/inter_10_percent_ce_2_geo_sin_add/epoch_10.pth'
pretrained = None
window_block_indexes = (
    list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) + list(range(12, 15)) + list(range(16, 19)) +
    list(range(20, 23)) + list(range(24, 27)))
residual_block_indexes = []

num_dec_layer = 6
lambda_2 = 2.0
data_root = '/lus/lfs1aip2/projects/u6ev/ahmad/datasets/hands23_data/'

model = dict(
    type='CoDETR',
    backbone=dict(
        type='ViT',
        img_size=640,
        pretrain_img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4*2/3,
        drop_path_rate=0.4,
        window_size=24,
        window_block_indexes=window_block_indexes,
        residual_block_indexes=residual_block_indexes,
        qkv_bias=True,
        use_act_checkpoint=True,
        init_cfg=None),
    neck=dict(        
        type='SFP',
        in_channels=[1024],        
        out_channels=256,
        num_outs=5,
        use_p2=True,
        use_act_checkpoint=False),
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
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0*num_dec_layer*lambda_2),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0*num_dec_layer*lambda_2)),
    query_head=dict(
        type='CoDINOHeadWithInteraction',
        num_query=1500,
        num_classes=3,
        num_feature_levels=5,
        in_channels=2048,
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
                with_cp=6, # number of layers that use checkpoint
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256, num_levels=5, dropout=0.0),
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
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        # --- DEFINE YOUR NEW LOSS COMPONENT HERE ---
        #loss_handside=dict(
        #type='CrossEntropyLoss',
        #use_sigmoid=False, # Must be False for standard CE with N classes
        #loss_weight=0.0    # The weight is now PART of the loss dict
        #),
        # --- ADD THIS DICTIONARY FOR THE INTERACTION LOSS ---
        loss_interaction=dict(
            type='FocalLoss',      # Use FocalLoss to handle class imbalance
            use_sigmoid=True,      # FocalLoss in MMDetection requires this
            gamma=2.0,             # Standard value for FocalLoss gamma
            alpha=0.25,            # Standard value for FocalLoss alpha
            loss_weight=3         # A good starting weight, tunable
        )
        ),
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
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0*num_dec_layer*lambda_2),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0*num_dec_layer*lambda_2)))],
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
            loss_weight=1.0*num_dec_layer*lambda_2),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0*num_dec_layer*lambda_2),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0*num_dec_layer*lambda_2)),],
    # model training and testing settings
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
            debug=False),],
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
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.



custom_imports = dict(
    imports=['mmdet.datasets.hands23_coco',
             'mmdet.core.hook.force_eval_mode_hook'],  # <-- path to your dataset file
    allow_failed_imports=False
)


#seed = 42
#deterministic = True
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadHandAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),  # or (640,640) for apples-to-apples
    # no RandomFlip, no AutoAugment, no RandomCrop
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleWithHand'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_handside', 'gt_interaction']),
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
    samples_per_gpu=32, #32, 4
    workers_per_gpu=32, #16, 4
    collate_fn=dict(type='custom_collate'),#custom_collate
    train=dict(filter_empty_gt=False, pipeline=train_pipeline,ann_file=data_root + 'annotations/val_h_first_second_full_corrected_w_area.json'), #val_h_first_second_full_10.json , val_h_first_second_full_corrected_w_area
    val=dict(pipeline=test_pipeline,ann_file=data_root + 'annotations/val_h_first_second_full_corrected_w_area.json'), #val_h_first_second_full_10.json, val_h_first_second_full_corrected_w_area.json
    test=dict(pipeline=test_pipeline,ann_file=data_root + 'annotations/val_h_first_second_full_corrected_w_area.json'),
    train_dataloader=dict(shuffle=False)
    )
'''
data = dict(
    samples_per_gpu=32, #32, 4
    workers_per_gpu=16, #16, 4
    collate_fn=dict(type='custom_collate'),#custom_collate
    train=dict(filter_empty_gt=False, pipeline=train_pipeline,ann_file=data_root + 'annotations/val_h_first_second_full.json'),
    val=dict(pipeline=test_pipeline,ann_file=data_root + 'annotations/val_h_first_second_full.json'),
    test=dict(pipeline=test_pipeline,ann_file=data_root + 'annotations/val_h_first_second_full.json'),
    train_dataloader=dict(shuffle=False)
    )

'''

'''

data = dict(
    samples_per_gpu=48,
    workers_per_gpu=16,
    collate_fn=dict(type='custom_collate'),#custom_collate
    train=dict(filter_empty_gt=False, pipeline=train_pipeline,ann_file=data_root + 'hd_epic_merged.json',
    img_prefix='/lus/lfs1aip2/home/s5a/ahmad.s5a/projects/hands23_data/hos_sequences_images_final0/'),
    val=dict(pipeline=test_pipeline,ann_file=data_root + 'hd_epic_merged.json',
    img_prefix='/lus/lfs1aip2/home/s5a/ahmad.s5a/projects/hands23_data/hos_sequences_images_final0/'),
    test=dict(pipeline=test_pipeline,ann_file=data_root + 'hd_epic_merged.json',
    img_prefix='/lus/lfs1aip2/home/s5a/ahmad.s5a/projects/hands23_data/hos_sequences_images_final0/')
    )


#train_dataloader=dict(shuffle=False)

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=12, 
    train=dict(filter_empty_gt=False, pipeline=train_pipeline,ann_file=data_root + 'annotations/train_h_first_second_full.json'),
    val=dict(pipeline=test_pipeline,ann_file=data_root + 'v1_valid_joint.json',
             img_prefix='/lus/lfs1aip2/home/s5a/ahmad.s5a/projects/hands23_data/finebio_object_detection_images/'),
    test=dict(pipeline=test_pipeline,ann_file=data_root + 'v1_valid_joint.json',
              img_prefix='/lus/lfs1aip2/home/s5a/ahmad.s5a/projects/hands23_data/finebio_object_detection_images/')
            )

data = dict(
    samples_per_gpu=48,
    workers_per_gpu=8,
    collate_fn=dict(type='custom_collate'),#custom_collate
    train=dict(filter_empty_gt=False, pipeline=train_pipeline,ann_file=data_root + 'HOIST/hoist_as_hands23_firstobject_val_trim_long.json',
    img_prefix='/lus/lfs1aip2/home/s5a/ahmad.s5a/projects/hands23_data/HOIST/valid/JPEGImages/'),
    val=dict(pipeline=test_pipeline,ann_file=data_root + 'HOIST/hoist_as_hands23_firstobject_val_trim_long.json',
    img_prefix='/lus/lfs1aip2/home/s5a/ahmad.s5a/projects/hands23_data/HOIST/valid/JPEGImages/'),
    test=dict(pipeline=test_pipeline,ann_file=data_root + 'HOIST/hoist_as_hands23_firstobject_val_trim_long.json',
    img_prefix='/lus/lfs1aip2/home/s5a/ahmad.s5a/projects/hands23_data/HOIST/valid/JPEGImages/')
    )
'''
evaluation = dict(metric='bbox')

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=1)

# optimizer
# We use layer-wise learning rate decay, but it has not been implemented.
# optimizer = dict(
#     type='AdamW',
#     lr=5e-5,
#     weight_decay=0.01,
#     constructor='LayerDecayOptimizerConstructor',
#     paramwise_cfg=dict(
#         num_layers=24, layer_decay_rate=0.8))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
optimizer = dict(
    type='AdamW',
    lr=0,
    weight_decay=0.01,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=24, layer_decay_rate=0.8))



custom_hooks = [
    dict(type='ForceEvalModeHook'),
    dict(
        type='ExpMomentumEMAHook',
        momentum=0.0001,
        priority=49),]

