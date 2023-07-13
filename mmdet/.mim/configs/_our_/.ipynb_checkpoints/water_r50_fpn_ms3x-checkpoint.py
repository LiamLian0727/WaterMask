_base_ = r'water_r50_fpn_1x.py/'

model = dict(
    roi_head=dict(
        mask_head=dict(
            num_heads_in_gat=2
        )
    )
)

dataset_type = 'CocoDataset'
data_root = 'data/UDW/'
img_norm_cfg = dict(
    mean = [81.236, 113.761, 117.095], std = [60.598, 58.471, 62.821], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 800), (1333, 768), (1333, 736), (1333, 704), (1333, 672), (1333, 640), 
                   (1333, 608), (1333, 576), (1333, 544), (1333, 512), (1333, 480)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(train=dict(pipeline=train_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)