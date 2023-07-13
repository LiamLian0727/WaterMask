import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from warnings import warn
from mmcv.ops import SimpleRoIAlign
from torch.nn.modules.utils import _pair

from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn import ConvModule, build_upsample_layer
from .fcn_mask_head import BYTES_PER_FLOAT, GPU_MEM_LIMIT, _do_paste_mask


def adj_index(h, k, node_num):
    dist = torch.cdist(h, h, p=2)
    each_adj_index = torch.topk(dist, k, dim=2).indices
    adj = torch.zeros(
        h.size(0), node_num, node_num, 
        dtype=torch.int, device=h.device, requires_grad = False
    ).scatter_(dim=2, index=each_adj_index, value=1)
    return adj


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        self.activation = nn.ELU()
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  
        e = self._prepare_attentional_mechanism_input(Wh)
        attention = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
        attention = F.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return self.activation(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, in_feature, out_feature,
                 top_k=11, token=3, alpha=0.2, num_heads=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.top_k = top_k
        hidden_feature = in_feature 
        self.conv = ConvModule(in_feature, hidden_feature, token, stride=token)
        self.attentions = [
            GraphAttentionLayer(
                hidden_feature, hidden_feature, alpha=alpha, concat=True
            )for _ in range(num_heads)
        ]
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(
            hidden_feature * num_heads, out_feature, alpha=alpha, concat=False)

        self.deconv = build_upsample_layer(
            cfg=dict(type='deconv', 
                     in_channels=out_feature, out_channels=out_feature, 
                     kernel_size=token, stride=token)
        )
        self.activation = nn.ELU()
        self._init_weights()

    def _init_weights(self):
        for m in [self.deconv]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.conv(x)
        batch_size, in_feature, column, row = h.shape  # h (N,C,H,W)
        node_num = column * row
        h = h.view(
            batch_size, in_feature, node_num).permute(0, 2, 1)  # h (N,H*W,in_feature)
        adj = adj_index(h, self.top_k, node_num)

        h = torch.cat([att(h, adj) for att in self.attentions], dim=2)
        h = self.activation(self.out_att(h, adj))

        h = h.view(batch_size, column, row, -1).permute(0, 3, 1, 2)
        h = F.interpolate(
           self.deconv(h), x.shape[-2:], mode='bilinear', align_corners=True)
        return F.relu(h+x)


class Fusion(nn.Module):

    def __init__(self, feat_dim, dilations=[1, 3, 5]):
        super(Fusion, self).__init__()

        for idx, dilation in enumerate(dilations):
            self.add_module(
                f'dilation_conv_{idx + 1}',
                ConvModule(feat_dim, feat_dim, kernel_size=3, padding=dilation, dilation=dilation)
            )

        self.merge_conv = ConvModule(feat_dim, feat_dim, kernel_size=1, act_cfg=None)

    def forward(self, x):
        return self.merge_conv(
            self.dilation_conv_1(x) + 
            self.dilation_conv_2(x) +  
            self.dilation_conv_3(x) +   
            F.avg_pool2d(x, x.shape[-1])
        )


class Stage(nn.Module):

    def __init__(self,
                 gff_in_channel=256,
                 gff_out_channel=256,

                 lcf_in_channel=256,
                 lcf_out_channel=256,

                 mask_out_size=14,
                 num_classes=7,
                 gff_out_stride=4,
                 upsample_cfg=dict(type='bilinear', scale_factor=2)):
        super(Stage, self).__init__()

        self.gff_out_stride = gff_out_stride
        self.num_classes = num_classes

        # for extracting gff branch feats
        self.gff_transform_in = nn.Conv2d(gff_in_channel, gff_out_channel, 1)
        self.gff_roi_extractor = SimpleRoIAlign(output_size=mask_out_size, spatial_scale=1.0 / gff_out_stride)

        fuse_in_channel = lcf_in_channel + gff_out_channel + 1
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(fuse_in_channel, lcf_in_channel, 1),
            Fusion(lcf_in_channel)
        ])

        self.fuse_transform_out = nn.Conv2d(lcf_in_channel, lcf_out_channel - 1, 1)
        self.upsample = build_upsample_layer(upsample_cfg.copy())
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in [self.gff_transform_in, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, lcf_feats, lcf_logits, gff_feat, rois, upsample=True):
        # gff-branch feats
        gff_feat = self.relu(self.gff_transform_in(gff_feat))
        ins_gff_feats = self.gff_roi_extractor(gff_feat, rois)

        concat_tensors = [lcf_feats, ins_gff_feats, lcf_logits.sigmoid()]
        fused_feats = torch.cat(concat_tensors, dim=1)
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))

        fused_feats = self.relu(self.fuse_transform_out(fused_feats))
        fused_feats = torch.cat([fused_feats, lcf_logits.sigmoid()], dim=1)

        fused_feats = self.upsample(fused_feats) if upsample else fused_feats
        return fused_feats


@HEADS.register_module()
class WaterMaskHead(nn.Module):

    def __init__(self,
                 num_convs_gff=2,
                 conv_in_channels_gff=256,
                 conv_out_channels_gff=256,
                 conv_kernel_size_gff=3,
                 
                 use_gat = True,
                 image_patch_token=3,
                 graph_top_k=11,
                 num_heads_in_gat=1,

                 num_convs_lcf=2,
                 conv_in_channels_lcf=256,
                 conv_out_channels_lcf=256,
                 conv_kernel_size_lcf=3,

                 conv_cfg=None,
                 norm_cfg=None,

                 gff_out_stride=4,
                 classes_num_in_stages=[7, 7, 1],
                 stage_output_mask_size=[14, 28, 56],
                 pre_upsample_last_stage=False,
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 loss_cfg=dict(type='LaplacianCrossEntropyLoss')):
        super(WaterMaskHead, self).__init__()

        self.num_convs_lcf = num_convs_lcf
        self.conv_kernel_size_lcf = conv_kernel_size_lcf
        self.conv_in_channels_lcf = conv_in_channels_lcf
        self.conv_out_channels_lcf = conv_out_channels_lcf
        self.build_layer('lcf')

        self.num_convs_gff = num_convs_gff
        self.conv_kernel_size_gff = conv_kernel_size_gff
        self.conv_in_channels_gff = conv_in_channels_gff
        self.conv_out_channels_gff = conv_out_channels_gff
        self.use_gat = use_gat
        self.image_patch_token=image_patch_token
        self.graph_top_k=graph_top_k
        self.num_heads_in_gat=num_heads_in_gat
        self.build_layer('gff')
        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.gff_out_stride = gff_out_stride
        self.stage_output_mask_size = stage_output_mask_size
        self.classes_num_in_stages = classes_num_in_stages
        self.num_classes = classes_num_in_stages[0]

        self.pre_upsample_last_stage = pre_upsample_last_stage
        self.loss_lcf = build_loss(loss_cfg)

        assert len(self.stage_output_mask_size) > 1
        self.stages = nn.ModuleList()
        out_channel = conv_out_channels_lcf
        stage_out_channels = [conv_out_channels_lcf]

        for idx, out_size in enumerate(self.stage_output_mask_size[:-1]):
            in_channel = out_channel
            out_channel = in_channel // 2

            new_stage = Stage(
                gff_in_channel=conv_out_channels_gff,
                gff_out_channel=in_channel,
                lcf_in_channel=in_channel,
                lcf_out_channel=out_channel,
                mask_out_size=out_size,
                num_classes=self.classes_num_in_stages[idx],
                gff_out_stride=gff_out_stride,
                upsample_cfg=upsample_cfg)

            self.stages.append(new_stage)
            stage_out_channels.append(out_channel)

        self.stage_lcf_logits = nn.ModuleList([
            nn.Conv2d(
                stage_out_channels[idx],
                num_classes,
                1) for idx, num_classes in enumerate(self.classes_num_in_stages)
        ])

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.stage_lcf_logits:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def build_layer(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(
                self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = ConvModule(
                in_channels, out_channels, conv_kernel_size, padding=1)
            convs.append(conv)
        if name == 'gff' and self.use_gat:
            convs.append(
                GAT(in_feature=out_channels, out_feature=out_channels,
                    top_k=self.graph_top_k, token=self.image_patch_token, num_heads=self.num_heads_in_gat))
        self.add_module(f'{name}_convs', nn.ModuleList(convs))

    def forward(self, lcf_feats, gff_feat, rois, roi_labels):
        for conv in self.lcf_convs:
            lcf_feats = conv(lcf_feats)
        
        for conv in self.gff_convs:
            gff_feat = conv(gff_feat)

        stage_lcf_preds = []
        for idx, stage in enumerate(self.stages):
            lcf_logits = self.stage_lcf_logits[idx](lcf_feats)[torch.arange(len(rois)), roi_labels][:, None]
            upsample_flag = self.pre_upsample_last_stage or idx < len(self.stages) - 1
            lcf_feats = stage(
                lcf_feats, lcf_logits, gff_feat, rois, upsample_flag)
            stage_lcf_preds.append(lcf_logits)

        # if use class-agnostic classifier for the last stage
        if self.classes_num_in_stages[-1] == 1:
            roi_labels = roi_labels.clamp(max=0)

        lcf_preds = self.stage_lcf_logits[-1](lcf_feats)[torch.arange(len(rois)), roi_labels][:, None]
        if not self.pre_upsample_last_stage:
            lcf_preds = F.interpolate(
                lcf_preds, scale_factor=2, mode='bilinear', align_corners=True)
        stage_lcf_preds.append(lcf_preds)

        return stage_lcf_preds

    def get_targets(self, pos_bboxes_list, pos_assigned_gt_inds_list, gt_masks_list):

        def resize_masks_as_stages_targets(
            proposals, assigned_gt_inds, gt_masks, mask_size=None):
            proposals_np = proposals.cpu().numpy()
            proposals_np[:, [0, 2]] = np.clip(
                proposals_np[:, [0, 2]], 0, gt_masks.width)
            proposals_np[:, [1, 3]] = np.clip(
                proposals_np[:, [1, 3]], 0, gt_masks.height)

            resize_masks = gt_masks.crop_and_resize(
                proposals_np,
                _pair(mask_size),
                inds=assigned_gt_inds.cpu().numpy(),
                device=proposals.device
            ).to_ndarray()

            return torch.from_numpy(resize_masks).float().to(proposals.device)

        stage_lcf_targets_list = [[] for _ in range(len(self.stage_output_mask_size))]
        for pos_bboxes, pos_assigned_gt_inds, gt_masks in zip(pos_bboxes_list,
                                                              pos_assigned_gt_inds_list,
                                                              gt_masks_list):
            stage_lcf_targets = [
                resize_masks_as_stages_targets(
                    pos_bboxes,
                    pos_assigned_gt_inds,
                    gt_masks,
                    mask_size=mask_size) for mask_size in self.stage_output_mask_size
            ]
            for stage_idx in range(len(self.stage_output_mask_size)):
                stage_lcf_targets_list[stage_idx].append(stage_lcf_targets[stage_idx])

        return [torch.cat(targets) for targets in stage_lcf_targets_list]

    def loss(self, stage_lcf_preds, stage_lcf_targets):
        loss_lcf = self.loss_lcf(stage_lcf_preds, stage_lcf_targets)
        return dict(loss_lcf=loss_lcf)

    '''
    This function is come from 
    mmdet.models.roi_heads.mask_heads.fcn_mask_head.py
    and has some change
    '''

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_pred = det_bboxes.new_tensor(mask_pred)
        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        # In most cases, scale_factor should have been
        # converted to Tensor when rescale the bbox
        if not isinstance(scale_factor, torch.Tensor):
            if isinstance(scale_factor, float):
                scale_factor = np.array([scale_factor] * 4)
                warn('Scale_factor should be a Tensor or ndarray '
                     'with shape (4,), float would be deprecated. ')
            assert isinstance(scale_factor, np.ndarray)
            scale_factor = torch.Tensor(scale_factor)

        if rescale:
            img_h, img_w = ori_shape[:2]
            bboxes = bboxes / scale_factor.to(bboxes)
        else:
            w_scale, h_scale = scale_factor[0], scale_factor[1]
            img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32)
            img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT)
            )
            assert (num_chunks <= N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8
        )

        if mask_pred.shape[1] > 1:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu'
            )

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds,) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms
