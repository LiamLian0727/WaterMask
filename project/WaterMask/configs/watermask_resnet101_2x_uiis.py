## ---------------------- DEFAULT_SETTING ----------------------
_base_ = [
    'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
    './uiis_dataset.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

default_scope = 'mmdet'
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', max_keep_ckpts=1, save_best=['coco/segm_mAP'], rule='greater', save_last=True
    ),
)

custom_imports = dict(imports=['project.WaterMask.watermask'], allow_failed_imports=False)


## ---------------------- MODEL_SETTING -------------------------

num_classes=len({{_base_.CLASSES}})

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
    ),
    roi_head=dict(
        type='WaterRoIHead',
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(
            _delete_=True,
            type='WaterMaskHead',
            num_convs_gff=2,
            num_convs_lcf=2,
            image_patch_token=3,
            graph_top_k=11,
            num_heads_in_gat=1,
            classes_num_in_stages=[num_classes, num_classes, 1],
            stage_output_mask_size=[14, 28, 56],
            loss_cfg=dict(
                type='LaplacianCrossEntropyLoss',
                stage_lcf_loss_weight=[0.25, 0.65, 1],
                boundary_width=3,
                start_stage=2)
        )
    )
)

## ---------------------- OPTIM_SETTING --------------------------

# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)
