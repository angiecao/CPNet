# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..utils import resize, Upsample
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
'''
@HEADS.register_module()
class PFlow_FPN_Head(BaseCascadeDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, **kwargs):
        super(PFlow_FPN_Head, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.conv_last = ConvModule(
                        len(self.in_channels) * self.channels,
                        self.channels,
                        3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)

    def forward(self, inputs_, prev_output):
        fpn_feature_list = prev_output
        fusion_list = [fpn_feature_list[3]]
        output_size = fpn_feature_list[3].shape[2:]
        for i in range(0, len(fpn_feature_list)-1):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        output_join_edge = self.cls_seg(x)
        #np.save("output_join_edge.npy", output_join_edge.data.cpu().numpy())
        return output_join_edge
'''
@HEADS.register_module()
class PFlow_FPN_Head(BaseCascadeDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super(PFlow_FPN_Head, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.in_channels[i] // 2,
                        self.in_channels[i] // 2,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs_, prev_output):
        fpn_feature_list = prev_output
        output = [self.scale_heads[0](fpn_feature_list[-1])]
        for i in reversed(range(len(fpn_feature_list) - 1)):
            # non inplace
            output.append(resize(
                self.scale_heads[3 - i](fpn_feature_list[i]),
                size=fpn_feature_list[-1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners))
        output = self.cls_seg(torch.cat(output, dim=1))
        return output

    def loss(self, inputs, prev_output, batch_data_samples, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, prev_output)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        #file_names = [img['filename'] for img in img_metas]

        #np.save("file_names.npy", file_names)

        #for i in range(4):
            #np.save("gt_seg{}.npy".format(i), gt_semantic_seg[1][i].data.cpu().numpy())
        return losses, seg_logits