# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[120.476, 81.193,  81.799],
    std=[54.847, 37.918, 39.321],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    test_cfg=dict(size=(512, 512)))
model = dict(
    type='CascadeEncoderDecoder3',
    data_preprocessor=data_preprocessor,
    num_stages=3,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        #这个out_indices很重要，就是可以把这个数组对应的stage索引的输出都保存下来
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='PFlowNeck',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=False)
    ),
    decode_head=[
        dict(
            type='PFNet_FaEdge_Head',
            in_channels=[256, 256, 256, 256],
            channels=64,
            in_index=[0, 1, 2, 3],
            dropout_ratio=0.1,
            num_classes=0,
            points_num=[128, 64, 32],
            use_sobel=False,
            use_scale=True,
            use_transpoint=False,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='EdgeSegLightLoss', edge_method='sobel', loss_weight=5.0)
        ),

        dict(
            type='PFlow_FPN_Head',
            feature_strides=[4, 8, 16, 32],
            in_channels=[256, 256, 256, 256],
            channels=512,
            in_index=[0, 1, 2, 3],
            dropout_ratio=0.1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

        dict(
            type='EdgePoint_Head',
            in_channels=256,
            points_num=512,
            use_scale=True,
            channels=256,
            num_fcs=3,
            num_classes=6,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=False),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
    #test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(320, 320)))
