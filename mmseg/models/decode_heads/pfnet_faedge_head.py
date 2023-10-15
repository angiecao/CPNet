import numpy as np
from torch import nn, einsum, Tensor
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, Sequential
from ..builder import HEADS
from .cascade_decode_head import BaseDecodeHead
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from ..losses import accuracy
from einops import repeat
from mmseg.utils import ConfigType, SampleList
from typing import List, Tuple

import warnings
from mmcv.ops import DeformConv2dPack as DCN
from ..utils import resize

warnings.filterwarnings("ignore")

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
        random_points = dict({32:64, 64: 128, 128: 512, 256: 1024, 512: 4096})
        point_indice_b = torch.topk(uncertainty_map.view(R, H * W), k=random_points[num_points], dim=1)[1]
        select_index = (torch.rand(num_points, device=uncertainty_map.device) * random_points[num_points]).long()
        point_indices = torch.index_select(point_indice_b, 1, select_index)
    else:
        point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    #point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    #point_coords[:, :, 0] = (point_indices % W).to(torch.float) * w_step
    #point_coords[:, :, 1] = (point_indices // W).to(torch.float) * h_step
    return point_indices
    '''
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    #select_index = (torch.rand(num_points, device=uncertainty_map.device) * (num_points*4)).int()
    #point_indices = torch.index_select(point_indice_b, 1, select_index)
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords
    '''

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

def point_sample(input, point_indices, use_neighbor=True, **kwargs):
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
    '''
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)

    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    '''
    return output

def exists(val):
    return val is not None


def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
        num_neighbors = None
    ):
        super().__init__()
        self.num_neighbors = num_neighbors

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v= nn.Linear(dim, dim, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim*attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim*attn_mlp_hidden_mult, dim),
        )

    def forward(self, x,pos):
        n, num_neighbors = x.shape[1], self.num_neighbors

        # get queries, keys, values
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        # expand values
        v = repeat(v, 'b j d -> b i j d', i = n)

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # attention
        attn = sim.softmax(dim = -2)

        agg = einsum('b i j d,b i j d -> b i d',attn,v)

        return agg

def make_sobel_kernel(kernel, in_planes, out_planes, theta):
    theta = torch.tensor(theta)
    sobel_kernel = torch.tensor(kernel).float()
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = Variable(sobel_kernel).repeat(out_planes, in_planes, 1, 1)
    sobel_kernel = sobel_kernel * theta.view(-1, 1, 1, 1)

    return sobel_kernel

class SobelEdge_Simple(nn.Module):
    def __init__(self, in_planes, norm_cfg, alpha=0.5, sigma=4):
        super(SobelEdge_Simple, self).__init__()
        self.in_planes = in_planes
        self.alpha = alpha
        self.sigma = sigma
        self.theta = 1.0
        self.sobel_kernel0 = make_sobel_kernel([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], in_planes, in_planes//4, self.theta)
        self.sobel_kernel90 = make_sobel_kernel([[1, 0, -1], [2, 0, -2], [1, 0, -1]], in_planes, in_planes//4, self.theta)
        self.sobel_kernel45 = make_sobel_kernel([[2, 1, 0], [1, 0, -1], [0, -1, -2]], in_planes, in_planes//4, self.theta)
        self.sobel_kernel135 = make_sobel_kernel([[0, -1, -2], [1, 0, -1], [2, 1, 0]], in_planes, in_planes//4, self.theta)
        self.edge_final = Sequential(
            ConvModule(
                in_planes,
                in_planes,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=None),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=1, padding=0, bias=False)
        )

    def forward(self,x):
        device = x.device
        edge_0 = F.conv2d(x, self.sobel_kernel0.to(device), stride=1, padding=1)
        edge_90 = F.conv2d(x, self.sobel_kernel90.to(device), stride=1, padding=1)
        edge_45 = F.conv2d(x, self.sobel_kernel0.to(device), stride=1, padding=1)
        edge_135 = F.conv2d(x, self.sobel_kernel0.to(device), stride=1, padding=1)
        out_cat = torch.cat((torch.abs(edge_0), torch.abs(edge_90), torch.abs(edge_45), torch.abs(edge_135)), 1)
        out = self.edge_final(out_cat)
        return out


class Point_Enhance(nn.Module):
    def __init__(self, dim=256, pos_dim=64, hidden_mult=2, edge_points=64, use_scale=True):
        super(Point_Enhance, self).__init__()
        self.edge_points=edge_points
        self.use_scale=use_scale
        self.point_layer = PointTransformerLayer(dim=dim,
                                                 pos_mlp_hidden_dim=pos_dim,
                                                 attn_mlp_hidden_mult=hidden_mult)
    def forward(self, edge_pred, feature):
        _, C, H, W = feature.shape
        point_indices = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.edge_points, use_scale=True)
        sample_x = point_indices % H
        sample_y = point_indices // W
        point_coords = torch.zeros(_, self.edge_points, 2, dtype=torch.float, device=edge_pred.device)
        point_coords[:, :, 0] = sample_x / W
        point_coords[:, :, 1] = sample_y / H
        edge_feat = point_sample(feature, point_indices)
        edge_feat_aggr = self.point_layer(edge_feat.permute(0, 2, 1), point_coords).permute(0, 2, 1)
        point_indices = point_indices.unsqueeze(1).expand(-1, C, -1).long()
        final_low_feature = feature.reshape(_, C, H * W).scatter(2, point_indices, edge_feat_aggr).view(_, C, H, W)
        #np.save("edge_pred_3.npy", edge_pred.cpu().detach().numpy())
        #np.save("point_indices.npy", point_indices.cpu().numpy())
        #np.save("final_low_feature.npy", final_low_feature.cpu().detach().numpy())
        #np.save("ori_feature.npy", feature.cpu().detach().numpy())
        return final_low_feature

class PointFlowModule(BaseModule):
    def __init__(self, in_planes, use_scale, use_sobel, use_transpoint,
                 dim=64, maxpool_size=8, avgpool_size=8, edge_points=64,
                 norm_cfg=None):
        super(PointFlowModule, self).__init__()
        self.dim = dim
        self.use_scale = use_scale
        self.use_sobel = use_sobel
        self.use_transpoint = use_transpoint
        if self.use_sobel:
            self.sobel_edge = SobelEdge_Simple(in_planes, norm_cfg)
            self.edge_sigmoid = nn.Sigmoid()
        else:
            self.point_matcher = GenerateSalientMap(dim)
            self.down_h = nn.Conv2d(in_planes, dim, 1)
            self.down_l = nn.Conv2d(in_planes, dim, 1)
            self.avgpool_size = avgpool_size
            self.avg_pool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))
            self.edge_final = Sequential(
                ConvModule(
                    in_planes,
                    in_planes,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU')),
                ConvModule(
                    in_planes,
                    1,
                    3,
                    padding=1,
                    norm_cfg=None,
                    act_cfg=None)
            )

        if self.use_transpoint:
            self.point_layer = Point_Enhance(edge_points=edge_points, use_scale=self.use_scale)
        '''
        self.offset = ConvModule(
                in_planes * 2,
                in_planes,
                1,
                bias=False,
                norm_cfg=None,
                act_cfg=None)
        '''
        self.offset = nn.Conv2d(in_planes * 2, in_planes, 1, bias=False)
        # self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
        #                        extra_offset_mask=True)
        self.dcpack_L2 = DCN(in_planes * 2, in_planes, 3, stride=1, padding=1, dilation=1, deform_groups=8)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x_high, x_low = x
        N, C, H, W = x_low.shape
        if not self.use_sobel:
            x_high_embed = self.down_h(x_high)
            x_low_embed = self.down_l(x_low)

            certainty_map = self.point_matcher([x_high_embed, x_low_embed])
            avgpool_grid = self.avg_pool(certainty_map)
            _, _, map_h, map_w = certainty_map.size()
            avgpool_grid = F.interpolate(avgpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)

            # edge part
            x_high_edge = x_high - x_high * avgpool_grid
            edge_pred = self.edge_final(x_high_edge)
        else:
            edge_pred = self.sobel_edge(x_high)
            edge_pred_copy = edge_pred.detach()
            x_high_edge = self.edge_sigmoid(edge_pred_copy) * x_high

        x_high_edge = F.interpolate(x_high_edge, size=(H, W), mode='bilinear', align_corners=False)
        x_high = F.interpolate(x_high, size=(H, W), mode='bilinear', align_corners=False)
        if not self.use_sobel:
            avgpool_grid_low = F.interpolate(avgpool_grid, size=(H, W), mode='bilinear', align_corners=False)
            x_low_edge = x_low - x_low * avgpool_grid_low
        else:
            edge_pred_low = F.interpolate(edge_pred_copy, size=(H, W), mode='bilinear', align_corners=False)
            edge_pred_low = self.edge_sigmoid(edge_pred_low)
            x_low_edge = edge_pred_low * x_low

        offset = self.offset(torch.cat([x_low_edge, x_high_edge * 2], dim=1))
        #offset = self.offset(torch.cat([x_low, x_high * 2], dim=1))
        feat_cat = torch.cat([x_high, offset], dim=1)
        feat_align = self.relu(self.dcpack_L2(feat_cat))
        #print("feat_align + x_low_edge:{}".format((feat_align + x_low_edge).isnan().any()))
        #print("edge_pred:{}".format(edge_pred.isnan().any()))

        if self.use_transpoint:
            final_low_features = self.point_layer(edge_pred_low,feat_align)+x_low_edge
        else:
            final_low_features = feat_align + x_low_edge

        return final_low_features, edge_pred



class GenerateSalientMap(nn.Module):
    def __init__(self, in_channels):
        super(GenerateSalientMap,self).__init__()
        self.match_conv = nn.Conv2d(in_channels*2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_high, x_low = x
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        certainty = self.match_conv(torch.cat([x_high, x_low], dim=1))
        return self.sigmoid(certainty)


@HEADS.register_module()
class PFNet_FaEdge_Head(BaseDecodeHead):
    def __init__(self, points_num, use_sobel, use_scale, use_transpoint, **kwargs):
        super(PFNet_FaEdge_Head, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.fpn_out_align = nn.ModuleList()
        self.points_num = points_num
        self.use_sobel = use_sobel
        self.use_scale = use_scale
        self.use_transpoint = use_transpoint
        for i in range(len(self.in_channels)-1):
            self.fpn_out_align.append(PointFlowModule(
                in_planes=self.in_channels[i],
                dim=64,
                maxpool_size=8,
                avgpool_size=8,
                edge_points=self.points_num[i],
                use_scale=use_scale,
                use_sobel=use_sobel,
                use_transpoint=use_transpoint,
                norm_cfg=self.norm_cfg))
        self.fpn_out = nn.ModuleList()
        for i in range(len(self.in_channels)-1):
            self.fpn_out.append(nn.Sequential(
                ConvModule(
                    self.in_channels[i],
                    self.in_channels[i],
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None),
                nn.ReLU(),
            ))
        del self.conv_seg

    def forward(self, inputs: Tuple[Tensor]) -> dict:
        x = self._transform_inputs(inputs)

        edge_maps = []
        f = x[-1]
        fpn_feature_list = [f]
        for i in reversed(range(len(x) - 1)):
            conv_x = x[i]
            f, edge_pred = self.fpn_out_align[i]([f, conv_x])
            f = conv_x + f
            edge_maps.append(edge_pred)
            fpn_feature_list.append(self.fpn_out[i](f))
        return edge_maps, fpn_feature_list

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
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
        seg_logits, output = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses, [seg_logits[-1], output]

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
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
        seg_logits, output = self.forward(inputs)
        #losses = self.losses(seg_logits, gt_semantic_seg)
        return self.predict_by_feat(seg_logits[-1], batch_img_metas)


    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss."""
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        for i in range(len(seg_logits)):
            seg_logit = seg_logits[i]
            seg_logit = resize(
                input=seg_logit,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

            loss[self.loss_decode.loss_name+str(i)] = self.loss_decode(
                seg_logit,
                seg_label,
                ignore_index=self.ignore_index)
        return loss