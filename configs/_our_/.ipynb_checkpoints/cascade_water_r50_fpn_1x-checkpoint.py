_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
fp16 = dict(loss_scale=512.)
model = dict(
    roi_head=dict(
        type='CascadeHiRoIHead',
        stage_loss_weights=[1, 0.5, 0.25],
        mask_head=dict(
            _delete_=True,
            type='HiMaskHead',
            num_convs_gff=2,
            num_convs_lcf=2,
            classes_num_in_stages=[7, 7, 1],
            stage_output_mask_size=[14, 28, 56], 
            loss_cfg=dict(
                type='LaplacianCrossEntropyLoss',
                stage_lcf_loss_weight=[0.25, 0.65, 1],
                boundary_width=3,
                start_stage=1)
        )
    )
)

data = dict(samples_per_gpu=2, workers_per_gpu=2)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
evaluation = dict(metric=['bbox', 'segm'], classwise=True, interval=1)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8,11],
    gamma=0.1)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))