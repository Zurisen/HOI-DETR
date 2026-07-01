_base_ = [
    './co_dino_5scale_swin_tiny_coco_with_relation_only_all_losses_custom.py'
]

pretrained = None

model = dict(
    backbone=dict(
        type='Swin',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.3,
        pretrained=pretrained),
    neck=dict(in_channels=[96, 192, 384, 768]),
    query_head=dict(in_channels=768))