_base_=['mask_rcnn_r50_fpn_2x_coco.py']


model = dict(roi_head=dict(bbox_head=dict(num_classes=20),mask_roi_extractor=None,mask_head=None))

dataset_type='VOCDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='PASCAL_VOC2012/PASCALVOC2012/ImageSets/Main/trainval.txt',
        img_prefix='PASCAL_VOC2012/PASCALVOC2012/',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file='PASCAL_VOC2012/PASCALVOC2012/ImageSets/Main/trainval.txt',
        img_prefix='PASCAL_VOC2012/PASCALVOC2012/',
    ),
    test=dict(
        type=dataset_type,
        ann_file='PASCAL_VOC2012/PASCALVOC2012/ImageSets/Main/trainval.txt',
        img_prefix='PASCAL_VOC2012/PASCALVOC2012/',
    )
)
runner = dict(type='EpochBasedRunner', max_epochs=8)
optimizer = dict(type='SGD', lr=0.001)
lr_config = None
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
load_from='mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'