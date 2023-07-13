_base_ = 'solov2_r50_fpn_1x_coco.py'

img_norm_cfg = dict(
    mean = [81.236, 113.761, 117.095], std = [60.598, 58.471, 62.821], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 800), (1333, 768), (1333, 736), (1333, 704), (1333, 672), (1333, 640)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

evaluation = dict(metric=['segm'], classwise=True, interval=3)
checkpoint_config = dict(interval=3)
