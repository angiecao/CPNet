# dataset settings
dataset_type = 'PotsdamDataset'
data_root = '/home/caq/data/Vaihingen'
img_dir = 'data_overlap'
ann_dir = 'label_overlap'

crop_size = (512, 512)
img_scale = (600, 600)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', degree=90, prob=0.5),
    #dict(type='Pad', size=crop_size),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=img_dir, seg_map_path=ann_dir),
        ann_file='splits_overlap/train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        ann_file='splits_overlap/val.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = [dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore']),
                 dict(type='BoundaryMetric', iou_metrics=['BIoU'])]
test_evaluator = val_evaluator

img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [dict(type='LoadAnnotations')],[dict(type='PackSegInputs')]
        ])
]