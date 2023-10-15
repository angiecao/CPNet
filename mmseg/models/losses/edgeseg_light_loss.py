# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from ..builder import LOSSES
import numpy as np
from .utils import get_class_weight, weight_reduce_loss

def sobel_kernel(channel_in, channel_out, theta):
    sobel_kernel0 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel0 = sobel_kernel0.reshape((1, 1, 3, 3))
    sobel_kernel0 = torch.from_numpy(sobel_kernel0)
    sobel_kernel0 = sobel_kernel0.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel0 = sobel_kernel0*theta.view(-1, 1, 1, 1)

    sobel_kernel45 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel45 = sobel_kernel45.reshape((1, 1, 3, 3))
    sobel_kernel45 = torch.from_numpy(sobel_kernel45)
    sobel_kernel45 = sobel_kernel45.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel45 = sobel_kernel45*theta.view(-1, 1, 1, 1)

    sobel_kernel90 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')
    sobel_kernel90 = sobel_kernel90.reshape((1, 1, 3, 3))
    sobel_kernel90 = torch.from_numpy(sobel_kernel90)
    sobel_kernel90 = sobel_kernel90.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel90 = sobel_kernel90*theta.view(-1, 1, 1, 1)

    sobel_kernel135 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel135 = sobel_kernel135.reshape((1, 1, 3, 3))
    sobel_kernel135 = torch.from_numpy(sobel_kernel135)
    sobel_kernel135 = sobel_kernel135.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel135 = sobel_kernel135*theta.view(-1, 1, 1, 1)

    return sobel_kernel0, sobel_kernel45, sobel_kernel90, sobel_kernel135

@LOSSES.register_module()
class EdgeSegLightLoss(nn.Module):
    def __init__(self, dice=False, loss_name='loss_edge', edge_method='laplacian', loss_weight=1.0):
        super(EdgeSegLightLoss, self).__init__()
        self.dice_loss = dice
        # self.seg_loss = nn.CrossEntropyLoss(ignore_index=ignore_index).cuda()
        self._loss_name = loss_name
        self.loss_weight = loss_weight
        self.edge_method = edge_method
        if self.edge_method=='sobel':
            self.sobel_kernels = sobel_kernel(1,1,torch.tensor(2.0))
        # self.criterion = F.binary_cross_entropy_with_logits

    def bce2d(self, input, target):
        """
        For edge
        """
        target = target.unsqueeze(1)
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num

        weight = torch.Tensor(log_p.size()).fill_(0).to(log_p.device)
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num
        weight[ignore_index] = 0
        weight_gpu = weight.to(log_p.device)

        loss = F.binary_cross_entropy_with_logits(log_p, target_t.float(), weight_gpu, size_average=True)
        return loss

    def get_edges_lpc(self, targets, ignore_index):
        b = targets.shape[0]
        lpc_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).to(targets.device)
        lpc_kernel = lpc_kernel.reshape((1, 1, 3, 3))

        x = targets
        y1 = F.conv2d(x.float(), lpc_kernel, stride=1, padding=1)
        y2 = F.conv2d(x.float(), lpc_kernel, stride=2, padding=1)
        y3 = F.conv2d(x.float(), lpc_kernel, stride=4, padding=1)

        edge = torch.zeros_like(y1)
        y1[y1 < 0] = 1
        y1[y1 > 0] = 1
        edge = edge + y1

        y2 = F.interpolate(y2, scale_factor=2, mode='bilinear')
        y2[y2 < 0] = 1
        y2[y2 > 0] = 1
        edge = edge + y2

        y3 = F.interpolate(y3, scale_factor=4, mode='bilinear')
        y3[y3 < 0] = 1
        y3[y3 > 0] = 1
        edge = edge + y3

        edge[edge > 0] = 1
        edge[targets == ignore_index] = ignore_index
        edge = edge.squeeze(1)
        return edge

    def get_edges_sobel(self, targets, ignore_index):
        b = targets.shape[0]
        x = targets
        sobel_kernel0, sobel_kernel45, sobel_kernel90, sobel_kernel135 = \
            [kernel.to(targets.device) for kernel in self.sobel_kernels]
        y1 = F.conv2d(x.float(), sobel_kernel0.float(), stride=1, padding=1)
        y2 = F.conv2d(x.float(), sobel_kernel45.float(), stride=1, padding=1)
        y3 = F.conv2d(x.float(), sobel_kernel90.float(), stride=1, padding=1)
        y4 = F.conv2d(x.float(), sobel_kernel135.float(), stride=1, padding=1)
        edge = torch.zeros_like(y1)
        for y in [y1, y2, y3, y4]:
            y[y < 0] = 1
            y[y > 0] = 1
            edge += y
        edge[edge > 0] = 1
        edge[targets == ignore_index] = ignore_index
        return edge

    def forward(self,
                inputs,
                targets,
                ignore_index=255):
        if self.edge_method == 'laplacian':
            edge_targets = self.get_edges_lpc(targets, ignore_index)
        elif self.edge_method == 'sobel':
            edge_targets = self.get_edges_sobel(targets, ignore_index)
        edge_loss = self.bce2d(inputs, edge_targets)

        return edge_loss * self.loss_weight

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
