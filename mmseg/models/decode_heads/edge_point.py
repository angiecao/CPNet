import numpy as np
#import torch.nn as nn
from torch import nn, einsum
from mmcv.cnn import ConvModule

from ..utils import resize
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
import torch
from mmcv.ops import point_sample


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points, use_scale):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2
        ) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    if use_scale:
        #random_points = dict({32:64, 64: 128, 128: 512, 256: 1024, 512: 810})
        random_points = dict({32: 64, 64: 128, 128: 512, 256: 1024, 512: 4096})
        random_size = H*W//5
        point_indice_b = torch.topk(uncertainty_map.view(R, H * W), k=random_size, dim=1)[1]
        select_index = (torch.rand(num_points, device=uncertainty_map.device) * random_size).long()
        point_indices = torch.index_select(point_indice_b, 1, select_index)
        #point_indices = (torch.rand(num_points, device=uncertainty_map.device) * (H*W)).long()
        #point_indices = point_indices.unsqueeze(0).expand(R, -1)
    else:
        point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    #point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    #point_coords[:, :, 0] = (point_indices % W).to(torch.float) * w_step
    #point_coords[:, :, 1] = (point_indices // W).to(torch.float) * h_step
    return point_indices

def get_neigbor_indices(input, point_indices, choice=1):
    N, C, H, W = input.shape
    if choice==1:
        point_indices_new = torch.where((point_indices + 1) % W == 0, point_indices, point_indices+1)
    elif choice==2:
        point_indices_new = torch.where(point_indices % W == 0, point_indices, point_indices - 1)
    elif choice==3:
        point_indices_new = torch.where(point_indices < W, point_indices, point_indices - W)
    else:
        point_indices_new = torch.where(point_indices + W >= H*W, point_indices, point_indices + W)
    output = torch.gather(input.view(N, C, -1), dim=2, index=point_indices_new)

    return output

def point_sample(input, point_indices, use_neighbor=False, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    N,C,H,W = input.shape
    point_indices = point_indices.unsqueeze(1)
    point_indices= point_indices.expand(-1, C, -1)
    output = torch.gather(input.view(N,C,-1), dim=2, index=point_indices)
    if use_neighbor:
        output_right = get_neigbor_indices(input, point_indices, 1)
        output_left = get_neigbor_indices(input, point_indices, 2)
        output_up = get_neigbor_indices(input, point_indices, 3)
        output_down = get_neigbor_indices(input, point_indices, 4)
        output = torch.stack([output, output_right, output_left, output_up, output_down], dim=2)

    return output


@HEADS.register_module()
class EdgePoint_Head(BaseCascadeDecodeHead):
    def __init__(self,
                 points_num,
                 use_scale,
                 num_fcs,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=False),
                 **kwargs):
        super(EdgePoint_Head, self).__init__(**kwargs)
        self.points_num=points_num
        self.use_scale=use_scale
        self.cat_neighbor = nn.Conv1d(self.in_channels*5,self.in_channels,1)
        self.fcs = nn.ModuleList()
        self.num_fcs=num_fcs
        fc_in_channels = self.in_channels+self.num_classes
        for k in range(num_fcs):
            fc = ConvModule(
                fc_in_channels,
                self.channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.fcs.append(fc)
            fc_in_channels = self.channels + self.num_classes
        self.aux_point_conv = nn.Conv1d(self.channels, self.num_classes, 1)

    def forward(self, out_cat, coarse_feat):


        new_feat = self.cat_neighbor(out_cat)
        x = torch.cat([new_feat, coarse_feat], dim=1)
        for i,fc in enumerate(self.fcs):
            x = fc(x)
            if i!=self.num_fcs-1:
                x = torch.cat((x, coarse_feat), dim=1)
        return self.aux_point_conv(x)

    def forward_flops(self, inputs, prev_output, test_cfg):
        edge_pred, last_feature, fpn_logit = prev_output
        N, C, H, W = last_feature.shape
        point_indices = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.points_num,
                                                           use_scale=False)
        edge_feat_neighbor = point_sample(last_feature, point_indices, use_neighbor=True)
        coarse_feat = point_sample(fpn_logit, point_indices, use_neighbor=False)
        out_cat = edge_feat_neighbor.view(N, self.in_channels * 5, self.points_num)
        edge_logit = self.forward(out_cat, coarse_feat)
        point_indices = point_indices.unsqueeze(1).expand(-1, self.num_classes, -1).long()
        final_low_feature = fpn_logit.reshape(N, self.num_classes, H * W).scatter(2, point_indices, edge_logit).view(N, self.num_classes, H, W)
        return final_low_feature

    def loss(self, inputs, prev_output, batch_data_samples, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
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
        edge_pred, fined_pred, prev_pred = prev_output
        N, C, H, W = fined_pred.shape
        edge_pred = resize(
            input=edge_pred,
            size=fined_pred.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        with torch.no_grad():
            point_indices = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.points_num,
                                                               use_scale=self.use_scale)
        edge_feat_neighbor = point_sample(fined_pred, point_indices, use_neighbor=True)
        coarse_feat = point_sample(prev_pred, point_indices, use_neighbor=False)
        out_cat = edge_feat_neighbor.view(N, self.in_channels * 5, self.points_num)
        points_logits = self.forward(out_cat, coarse_feat)

        seg_label = self._stack_batch_gt(batch_data_samples)
        targets = resize(
            input=seg_label.float(),
            size=edge_pred.shape[-2:],
            mode='nearest')
        points_targets = point_sample(targets, point_indices, use_neighbor=False)
        losses = self.loss_by_feat(points_logits, points_targets)
        return losses

    def loss_by_feat(self, points_logits, points_targets):
        losses = dict()
        losses['point_loss'] = self.loss_decode(points_logits,
                                                   points_targets.squeeze(1).long(),
                                                   ignore_index=self.ignore_index)
        return losses

    def predict(self, inputs, prev_output, batch_data_samples, test_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
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
        edge_pred, last_feature, fpn_logit = prev_output
        N, C, H, W = last_feature.shape
        point_indices = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.points_num,
                                                               use_scale=False)
        edge_feat_neighbor = point_sample(last_feature, point_indices, use_neighbor=True)
        coarse_feat = point_sample(fpn_logit, point_indices, use_neighbor=False)
        out_cat = edge_feat_neighbor.view(N, self.in_channels * 5, self.points_num)
        edge_logit = self.forward(out_cat, coarse_feat)
        point_indices = point_indices.unsqueeze(1).expand(-1, self.num_classes, -1).long()
        final_low_feature = fpn_logit.reshape(N, self.num_classes, H * W).scatter(2, point_indices, edge_logit).view(N, self.num_classes, H, W)

        final_logit = resize(
                input=final_low_feature,
                size=batch_data_samples[0]['img_shape'],
                mode='bilinear',
                align_corners=self.align_corners)
        return final_logit