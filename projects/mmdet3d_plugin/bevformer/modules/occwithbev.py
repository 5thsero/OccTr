# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Qifang FU
# ---------------------------------------------
# TODO 不同尺寸特征融合

# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from torch.utils.checkpoint import checkpoint
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models import NECKS
from mmcv.cnn import normal_init
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner import BaseModule
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import ATTENTION

def project_points_on_img_v2(points, pts_range, bev_w, bev_h,
                             W_occ, H_occ,
                             return_depth=False):
    with torch.no_grad():
        # points: 2, wb*wb, 1, 1
        voxel_size = ((pts_range[2:] - pts_range[:2]) / torch.tensor([W_occ, H_occ])).to(points.device)
        points = points.permute(1, 2, 3, 0)  # wb*wb, 1, 1, 2
        points = points * voxel_size[None, None] + voxel_size / 2 + pts_range[:2][None, None].to(points.device)

        '''
        # project 3D point cloud (after bev-aug) onto multi-view images for corresponding 2D coordinates
        inv_bda = bda_mat.inverse()
        points = (inv_bda @ points.unsqueeze(-1)).squeeze(-1)

        # from lidar to camera
        points = points.view(-1, 1, 3)

        points = points - trans.reshape(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))

        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)'''

        # points_d = points[..., 2:3]
        points_uv = points[..., :2] # / (points_d + 1e-5)

        '''
        # from raw pixel to transformed pixel
        points_uv = post_rots[..., :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        '''

        points_uv[..., 0] = (points_uv[..., 0] / bev_w - 0.5) * 2
        points_uv[..., 1] = (points_uv[..., 1] / bev_h - 0.5) * 2

        mask = (points_uv[..., 0] > -1) & (points_uv[..., 0] < 1) \
               & (points_uv[..., 1] > -1) & (points_uv[..., 1] < 1)

    return points_uv.permute(2, 1, 0, 3), mask

@ATTENTION.register_module()
class OccwithBEV(BaseModule):  # img_bev_encoder_neck
    def __init__(self,
                 embed_dims=256,
                 norm_cfg=dict(type='BN2d'),
                 conv_cfg=dict(type='Conv2d'),
                 act_cfg=dict(type='ReLU'),
                 occ_thres=0.1,
                 num_frame=1,
                 lookback=[],
                 with_cp=False,
                 batch_first=True,
                 occ_h=200,
                 occ_w=200):
        super().__init__()
        # # conduct cam mask to reduce lookback space 相机掩码减少回看空间
        self.lookback = lookback
        self.upsample_cfg = dict(mode='trilinear')
        self.embed_dims = embed_dims
        # conv2d_cfg=dict(type='Conv2d')
        # norm2d_cfg=dict(type='BN', eps=1e-3, momentum=0.01)
        #self.in_channels = in_channels
        #self.out_channels = out_channels
        self.num_frame = num_frame
        self.occ_h = occ_h
        self.occ_w = occ_w

        self.in_c = 256
        self.mid_c = 256
        # self.upsample3d =  nn.Upsample(
        #     scale_factor=2, mode='trilinear', align_corners=True)
        # look back network module TODO
        if self.lookback != 0:  # 经过处理的图像特征与voxel 特征融合
            self.img_mlp = nn.Sequential(
                ConvModule(self.embed_dims, self.embed_dims,
                           kernel_size=1, padding=0, stride=1,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                           act_cfg=act_cfg, bias=False, inplace=True),
            )
            self.final_mlp = nn.Sequential(
                ConvModule(self.embed_dims+self.embed_dims, self.embed_dims,  #  + self.embed_dims
                           kernel_size=1, padding=0, stride=1,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                           act_cfg=act_cfg, bias=False, inplace=True),
            )

        self.upsample = nn.Sequential()
        self.upsample.add_module('upsample', nn.Identity())
        # 上采样到500*500
        self.upsample.add_module('interpolate', nn.Upsample(size=(200, 200), mode='bilinear', align_corners=False))

        self.lateral_convs = nn.ModuleList()
        # self.num_in = len(self.in_channels)
        '''只有一层occ因此不需要统一通道数   通道数256
        for i in range(self.num_in):
            l_conv = nn.Sequential(
                ConvModule(in_channels[i], out_channels,
                           kernel_size=1, padding=0,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                           act_cfg=act_cfg, bias=False,
                           inplace=True),
            )
            self.lateral_convs.append(l_conv)
        '''

        self.with_cp = with_cp

    def get_bev_feats(self, occ, prev_bev):  # laterals[0] bs, 256, D, H, W

        _, _, bev_h, bev_w = prev_bev.shape
        # transform = img_inputs[1:8]

        B, C, H, W = occ.shape

        # parse multi-lvl image features
        num_levels = 1
        lvl_feat_list = []
        lvl_list = []
        B_i, C_i, W_i, H_i = prev_bev.shape
        act_bs = B_i // 1
        lvl_list.append(prev_bev.reshape(act_bs, 1, C_i, W_i, H_i))  # 6个cam-->1
        lvl_feat = torch.cat(lvl_list, dim=1)  # lvl_id的所有帧图像特征
        lvl_feat_list.append(lvl_feat)  # fid-lvl_id ---> lvl_id - fid     bs ---> act_bs, 6
        # lvl_feat_list  (num_levels, num_frame, act_bs, 6, C_i, W_i, H_i)   (1,1,1,1,C_i, W_i, H_i)
        num_frame = 1

        mvs_feat = occ.new_zeros((B, num_levels * num_frame, C_i, W, H))
        coarse_coord_x, coarse_coord_y = torch.meshgrid(torch.arange(W).to(occ.device), torch.arange(H).to(occ.device))
        coord = torch.stack([coarse_coord_x, coarse_coord_y], dim=0)  # 2, W, H
        point_cloud_range = torch.Tensor([-40, -40, 40, 40])  # todo

        occ_mask = occ.new_ones((B, W, H), dtype=torch.bool)
        feat_list = []

        for b in range(act_bs):
            selected_coord = coord[:, occ_mask[b]].unsqueeze(-1).unsqueeze(-1)  # 2, wb, hb, 1, 1
            img_uv, img_mask = project_points_on_img_v2(
                selected_coord,
                pts_range=point_cloud_range, bev_w=bev_w, bev_h=bev_h,
                W_occ=W, H_occ=H
            )
            img_uv = selected_coord.squeeze(-1).permute(2,1,0).reshape(B, H, W, -1)
            # 2,wb*hb, 1, 1    wb*hb, 1, 1

            img_mask = img_mask.permute(2, 1, 0)[:, None].reshape(num_frame, 1, 1, -1, 1)
            # num_frame, 1, 1, wb*hb, 1
            for lvl_id in range(num_levels):
                sampled_img_feat = F.grid_sample(lvl_feat_list[lvl_id][b].float(),
                                                 img_uv.float(), align_corners=True, mode='bilinear',
                                                 padding_mode='zeros')  # 2， select, 1, 1
                sampled_img_feat = sampled_img_feat.reshape(num_frame, 1, C_i, -1,
                                                            1)  # default img channel    (num_frame, 6, C_i, select, 1, 1)
                sampled_img_feat = torch.sum(sampled_img_feat * img_mask, dim=1) / (
                            torch.sum(img_mask, dim=1) + 1e-12)  # (num_frame, C_i, select, 1, 1) 平均值
                mvs_feat[b, lvl_id * num_frame:(lvl_id + 1) * num_frame, :, occ_mask[b]] = sampled_img_feat.squeeze(-1)

        mvs_feat = mvs_feat.reshape(act_bs, num_frame * num_levels * C_i, W, H)
        # mvs_feat = self.upsample(prev_bev)
        mvs_feat = self.img_mlp(mvs_feat).permute(0, 1, 3, 2)

        occ = self.final_mlp(torch.cat([occ, mvs_feat], dim=1))
        # occ_with_bev = torch.empty((B, 2*C, H, W), device=prev_bev.device)
        # occ_with_bev[:, :C, ...] = occ
        # occ_with_bev[:, C:, ...] = mvs_feat
        # occ = self.final_mlp(occ_with_bev)
        return occ

    def forward(self, occ, prev_bev=None, bev_h=None,bev_w=None,img_inputs=None, img_metas=None, do_history=False):
        # first convert all x_s into the same dimention
        '''
        occ: occ特征
        prev_bev
        '''
        '''只有一层occ因此不需要统一通道数   通道数256
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            lateral_i = lateral_conv(feats[i])
            laterals.append(lateral_i)  # 通道数 64，128，256--->256   H,W大小不变
        '''
        '''只有一层occ不需要统一尺寸  尺寸500*500
        # build down-top path
        for i in range(self.num_in - 1, 0, -1):  # 2， 1
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i],
                                                              size=prev_shape, align_corners=False,
                                                            **self.upsample_cfg)  # 上采样到laterals[0]并与laterals[0]相加
        '''
        bs,_,dim = occ.shape
        x = occ.reshape(bs, self.occ_h, self.occ_w, dim).permute(0, 3, 1, 2)
        prev_bev = prev_bev.reshape(bs, bev_h, bev_w, dim).permute(0, 3, 1, 2)  # 1，c, h, w
        #if self.lookback != []:
        x = self.get_bev_feats(x, prev_bev)  # 把laterals[0]送入回看
        x = F.interpolate(x, size=(bev_h, bev_w), mode='bilinear', align_corners=False)
        x = x.reshape(bs, dim, -1).permute(0, 2, 1)

        return x


