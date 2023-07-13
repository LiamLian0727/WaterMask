import torch
import warnings
import numpy as np
import torch.nn.functional as F

from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.losses.cross_entropy_loss import generate_block_target


@HEADS.register_module()
class WaterRoIHead(StandardRoIHead):
    
    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois, torch.LongTensor([0]).to('cuda'))
            outs = outs + (mask_results['stage_lcf_preds'], )
        return outs

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_metas):
        """Run forward function and calculate loss for mask head in training."""

        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]
        
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_rois = bbox2roi(pos_bboxes)

        mask_results = self._mask_forward(x, pos_rois, torch.cat(pos_labels))
        stage_mask_targets = self.mask_head.get_targets(pos_bboxes, pos_assigned_gt_inds, gt_masks)
        loss_mask = self.mask_head.loss(mask_results['stage_lcf_preds'], stage_mask_targets)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def _mask_forward(self, x, rois, roi_labels):
        """Mask head forward function used in both training and testing."""

        ins_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        stage_lcf_preds = self.mask_head(ins_feats, x[0], rois, roi_labels)
        return dict(stage_lcf_preds=stage_lcf_preds)
 

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""

        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        #         if det_bboxes.shape[0] == 0:
        #             segm_result = [[] for _ in range(self.mask_head.classes_num_in_stages[0])]
        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [
                [[] for _ in range(self.mask_head.num_classes)] for _ in range(num_imgs)
            ]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
                #                 scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
                #             _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            _bboxes = [
                det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois, torch.cat(det_labels))

            stage_lcf_preds = mask_results['stage_lcf_preds'][self.mask_head.loss_lcf.start_stage-1:]
            for idx in range(len(stage_lcf_preds)-1):
                lcf_pred = stage_lcf_preds[idx].squeeze(1).sigmoid() >= 0.5
                non_boundary_mask = (generate_block_target(
                    lcf_pred, boundary_width=self.mask_head.loss_lcf.boundary_width+1) != 1).unsqueeze(1)
                non_boundary_mask = F.interpolate(
                    non_boundary_mask.float(),
                    stage_lcf_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
                pre_pred = F.interpolate(
                    stage_lcf_preds[idx],
                    stage_lcf_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
                stage_lcf_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
            lcf_pred = stage_lcf_preds[-1]
            
             # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            lcf_pred = lcf_pred.split(num_mask_roi_per_img, 0)

            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append([[] for _ in range(self.mask_head.classes_num_in_stages[0])])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        lcf_pred[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)
        return segm_results
