import torch
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_plugin_layer

from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module()
class R3MaskHead(FCNMaskHead):

    def __init__(self, with_conv_res=True, nl_stages=None, nl_cfg=None,
                 *args, **kwargs):
        super(R3MaskHead, self).__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        # remove the if because it's mandatory
        if self.with_conv_res:
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

        if nl_cfg is not None:
            self.nl_cfg = nl_cfg.copy() if nl_cfg is not None else None
            self.nl_stages = nl_stages or (False, False)

            if self.nl_stages[0]:
                # add nl layer
                self.nl_cfg['in_channels'] = kwargs['in_channels']
                self.convs.insert(0, build_plugin_layer(
                    self.nl_cfg, '_fcs_pre')[1])

            if self.nl_stages[1]:
                # add nl layer
                self.nl_cfg['in_channels'] = kwargs['conv_out_channels']
                self.convs.append(build_plugin_layer(
                    self.nl_cfg, '_fcs_post')[1])

    def init_weights(self):
        super(R3MaskHead, self).init_weights()
        # remove the if because it's mandatory
        if self.with_conv_res:
            self.conv_res.init_weights()

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        if res_feat is None:
            res_feat = torch.zeros_like(x)

        res_feat = self.conv_res(res_feat)
        x = x + res_feat

        for conv in self.convs:
            x = conv(x)
        res_feat = x
        outs = []
        if return_logits:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
            mask_pred = self.conv_logits(x)
            outs.append(mask_pred)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]
