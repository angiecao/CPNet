# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from torch import Tensor, nn
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class CascadeEncoderDecoder3(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor: OptConfigType = None,
                 pretrained=None,
                 init_cfg=None):
        self.num_stages = num_stages
        super(CascadeEncoderDecoder3, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        #assert self.midtype in ['edge_map', 'edge_embd', 'edge_fusion']
        x = self.extract_feat(inputs)
        edge_pred, fpn_feature_list = self.decode_head[0].forward(x)
        coarse_pred = self.decode_head[1].forward(x, fpn_feature_list)

        prev = [edge_pred[-1], fpn_feature_list[-1], coarse_pred]
        final_logits = self.decode_head[2].predict(x, prev, batch_img_metas, self.test_cfg)

        return final_logits

    def _decode_head_forward_train(self, inputs: Tensor,
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode, prev_outputs = self.decode_head[0].loss(
            inputs, data_samples, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_0'))

        edge_pred, fpn_feature_list = prev_outputs

        loss_decode, coarse_pred = self.decode_head[1].loss(
            inputs, fpn_feature_list, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, f'decode_{1}'))

        prev = [edge_pred, fpn_feature_list[-1], coarse_pred]
        loss_decode = self.decode_head[2].loss(
            inputs, prev, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, f'decode_{2}'))

        return losses

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Run forward function and calculate loss for decode head in
        training."""
        x = self.extract_feat(inputs)
        edge_pred, fpn_feature_list = self.decode_head[0].forward(x)
        coarse_pred = self.decode_head[1].forward(x, fpn_feature_list)

        prev = [edge_pred[-1], fpn_feature_list[-1], coarse_pred]
        final_logits = self.decode_head[2].forward_flops(x, prev, self.test_cfg)

        return final_logits

