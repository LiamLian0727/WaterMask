import torch
import warnings
import numpy as np
import torch.nn.functional as F

from mmdet.models.builder import HEADS
from mmdet.core import bbox2result, bbox2roi, merge_aug_masks
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.models.losses.cross_entropy_loss import generate_block_target


@HEADS.register_module()
class CascadeWaterRoIHead(CascadeRoIHead):

    def _mask_forward(self, stage, x, rois, roi_labels):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        ins_feats = mask_roi_extractor(
            x[:mask_roi_extractor.num_inputs], rois)
        stage_lcf_preds = mask_head(ins_feats, x[0], rois, roi_labels)

        return dict(stage_lcf_preds=stage_lcf_preds)

    def _mask_forward_train(self, stage, x, sampling_results, gt_masks, rcnn_train_cfg, bbox_feats=None):
        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results]
        pos_rois = bbox2roi(pos_bboxes)

        mask_results = self._mask_forward(
            stage, x, pos_rois, torch.cat(pos_labels))
        stage_mask_targets = self.mask_head[stage].get_targets(
            pos_bboxes, pos_assigned_gt_inds, gt_masks)
        loss_mask = self.mask_head[stage].loss(
            mask_results['stage_lcf_preds'], stage_mask_targets)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[
                    [] for _ in range(mask_classes)
                ] for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[
                    [] for _ in range(mask_classes)
                ] for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(
                            scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(
                        i, x, mask_rois, torch.cat(det_labels))
                    stage_lcf_preds = mask_results['stage_lcf_preds'][1:]
                    for idx in range(len(stage_lcf_preds) - 1):
                        lcf_pred = stage_lcf_preds[idx].squeeze(1).sigmoid() >= 0.5
                        non_boundary_mask = (generate_block_target(
                            lcf_pred, 
                            boundary_width= self.mask_head[i].loss_lcf.boundary_width + 1
                        ) != 1).unsqueeze(1)
                        non_boundary_mask = F.interpolate(
                            non_boundary_mask.float(),
                            stage_lcf_preds[idx + 1].shape[-2:], 
                            mode='bilinear', align_corners=True) >= 0.5
                        pre_pred = F.interpolate(
                            stage_lcf_preds[idx],
                            stage_lcf_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
                        stage_lcf_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
                    lcf_pred = stage_lcf_preds[-1]

                    # split batch mask prediction back to each image
                    lcf_pred = lcf_pred.split(num_mask_rois_per_img, 0)

                    aug_masks.append([
                        m.sigmoid().cpu().detach().numpy() for m in lcf_pred
                    ])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results