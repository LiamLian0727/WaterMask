import torch.nn as nn
from torch.nn.modules.utils import _pair
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_plugin_layer

from mmdet.models.builder import HEADS
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class FCCBBoxHead(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self, nl_stages=None, nl_cfg=None, *args, **kwargs):
        self.nl_btw_cfg = nl_cfg.copy() if nl_cfg is not None else None
        self.nl_btw_stages = nl_stages or (False, False)
        super(FCCBBoxHead, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=False)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False, with_nl=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            in_channel = self.conv_out_channels
            out_channel = in_channel // 2
            k_size = 7
            p_size = 3
            if self.nl_btw_stages[0]:
                # add nl layer
                self.nl_btw_cfg['in_channels'] = in_channel
                branch_fcs.append(build_plugin_layer(
                    self.nl_btw_cfg, '_fcs_pre')[1])
            for i in range(num_branch_fcs):
                branch_fcs.append(ConvModule(
                        in_channel,
                        out_channel,
                        _pair(k_size),
                        padding=p_size,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                in_channel = out_channel
                out_channel = in_channel // 2

            last_layer_dim = in_channel * k_size * k_size
            if self.nl_btw_stages[1]:
                # add nl layer
                self.nl_btw_cfg['in_channels'] = in_channel
                branch_fcs.append(build_plugin_layer(self.nl_btw_cfg, '_fcs')[1])
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

            x = x.flatten(1)
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCCBBoxHead(FCCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)