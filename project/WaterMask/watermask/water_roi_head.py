from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList

from .cross_entropy_loss import generate_block_target


@MODELS.register_module()
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

    def mask_loss(self, x: Tuple[Tensor], sampling_results: List[SamplingResult],
                  bbox_feats: Tensor, batch_gt_instances: InstanceList) -> dict:
        """Perform forward propagation and loss calculation of the water mask
        head on the features of the upstream network."""

        pos_priors = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]

        num_pos = sum(label.numel() for label in pos_labels)
        if num_pos == 0:
            dummy_loss = x[0].sum() * 0
            for param in self.mask_head.parameters():
                dummy_loss = dummy_loss + param.sum() * 0
            for param in self.mask_roi_extractor.parameters():
                dummy_loss = dummy_loss + param.sum() * 0
            return dict(loss_mask=dict(loss_lcf=dummy_loss))

        pos_rois = bbox2roi(pos_priors)
        roi_labels = torch.cat(pos_labels)

        mask_results = self._mask_forward(x, pos_rois, roi_labels)
        gt_masks = [gt_instances.masks for gt_instances in batch_gt_instances]
        stage_mask_targets = self.mask_head.get_targets(
            pos_priors, pos_assigned_gt_inds, gt_masks)
        loss_mask = self.mask_head.loss(
            mask_results['stage_lcf_preds'], stage_mask_targets)

        mask_results.update(loss_mask=loss_mask, mask_targets=stage_mask_targets)
        return mask_results

    def _mask_forward(self, x: Tuple[Tensor], rois: Tensor,
                      roi_labels: Tensor = None):
        """Mask head forward function used in both training and testing."""

        if roi_labels is None:
            roi_labels = rois.new_zeros((rois.size(0), ), dtype=torch.long)

        ins_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        stage_lcf_preds = self.mask_head(ins_feats, x[0], rois, roi_labels)
        return dict(
            stage_lcf_preds=stage_lcf_preds,
            mask_preds=stage_lcf_preds[-1],
            mask_feats=ins_feats)
 
    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the water mask head and predict
        segmentation masks with mmdet3 InstanceData format."""

        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)

        det_labels = [res.labels for res in results_list]
        roi_labels = torch.cat(det_labels)
        mask_results = self._mask_forward(x, mask_rois, roi_labels)

        stage_lcf_preds = mask_results['stage_lcf_preds'][
            self.mask_head.loss_lcf.start_stage - 1:]
        for idx in range(len(stage_lcf_preds) - 1):
            lcf_pred = stage_lcf_preds[idx].squeeze(1).sigmoid() >= 0.5
            non_boundary_mask = (generate_block_target(
                lcf_pred,
                boundary_width=self.mask_head.loss_lcf.boundary_width + 1) !=
                                 1).unsqueeze(1)
            non_boundary_mask = F.interpolate(
                non_boundary_mask.float(),
                stage_lcf_preds[idx + 1].shape[-2:],
                mode='bilinear',
                align_corners=True) >= 0.5
            pre_pred = F.interpolate(
                stage_lcf_preds[idx],
                stage_lcf_preds[idx + 1].shape[-2:],
                mode='bilinear',
                align_corners=True)
            stage_lcf_preds[idx + 1][non_boundary_mask] = pre_pred[
                non_boundary_mask]

        lcf_pred = stage_lcf_preds[-1]
        num_mask_roi_per_img = [len(res) for res in results_list]
        lcf_pred = lcf_pred.split(num_mask_roi_per_img, 0)

        for i, results in enumerate(results_list):
            if len(results) == 0:
                continue

            segm_result = self.mask_head.get_seg_masks(
                lcf_pred[i],
                results.bboxes,
                results.labels,
                self.test_cfg,
                batch_img_metas[i]['ori_shape'],
                batch_img_metas[i]['scale_factor'],
                rescale)
            results.masks = self._segms_to_instance_masks(
                segm_result, results.labels, results.bboxes.device)

        return results_list

    def _segms_to_instance_masks(self, cls_segms: List[list], labels: Tensor,
                                 device: torch.device) -> Tensor:
        """Convert class-wise segmentation list to instance-ordered mask
        tensor."""
        class_cursor = [0 for _ in range(len(cls_segms))]
        ordered_masks = []

        for label in labels.detach().cpu().tolist():
            ordered_masks.append(torch.from_numpy(cls_segms[label][class_cursor[label]]))
            class_cursor[label] += 1

        return torch.stack(ordered_masks, dim=0).to(device=device, dtype=torch.bool)
