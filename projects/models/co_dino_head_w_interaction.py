# Copyright (c) OpenMMLab. All rights reserved.
#this one with co_dino_head_w_interaction! THIS IS THE MAIN ONE. IN THE PAPER
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, multi_apply,
                        reduce_mean, bbox_overlaps)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS
from mmcv.ops import batched_nms
from projects.models import CoDeformDETRHead
from projects.models.query_denoising import build_dn_generator
#from projects.models.roi_heads.hand_side_head import HandSideHead
from projects.models.roi_heads.interaction_head import InteractionHead

from mmcv.parallel import DataContainer as DC
from projects.models.utils.position_encoding import *
import math
@HEADS.register_module()
class CoDINOHeadWithInteraction(CoDeformDETRHead):

    def __init__(self,
                 *args,
                 num_query=900,
                 dn_cfg=None,
                 transformer=None,
                 loss_handside=None,
                 loss_interaction=None,
                 **kwargs):

        if 'two_stage_num_proposals' in transformer:
            assert transformer['two_stage_num_proposals'] == num_query, \
                'two_stage_num_proposals must be equal to num_query for DINO'
        else:
            transformer['two_stage_num_proposals'] = num_query
        super(CoDINOHeadWithInteraction, self).__init__(
            *args, num_query=num_query, transformer=transformer, **kwargs)

        assert self.as_two_stage, \
            'as_two_stage must be True for DINO'
        assert self.with_box_refine, \
            'with_box_refine must be True for DINO'
        
        '''
        self.hand_side_head = HandSideHead(
            in_dim=self.embed_dims, 
            hidden_dim=128, 
            num_hand_classes=2,  # left/right
            loss_handside=loss_handside
        )
        '''
        self.pos_encoding_emb_size = 256
        self.pe_sine = PositionEmbeddingSine(
        num_pos_feats=self.pos_encoding_emb_size,
        normalize=True,
        scale=2*math.pi,
        warmup_cache=False
        )   
        
        ##for sam pos encoding
        self.interaction_head = InteractionHead(
            in_dim=2*self.embed_dims,      # The dimension of the decoder queries + pos
            hidden_dim=256,              # Your chosen hidden dimension
            final_hidden_dim=256,        # Your chosen final hidden dimension
            loss_interaction=loss_interaction  # <-- Pass the entire config dict here
        )
        '''
        ##for 11 values pos encoding
        self.interaction_head = InteractionHead(
            in_dim=2*self.embed_dims  + 11,      # The dimension of the decoder queries + pos
            hidden_dim=256,              # Your chosen hidden dimension
            final_hidden_dim=256,        # Your chosen final hidden dimension
            loss_interaction=loss_interaction  # <-- Pass the entire config dict here
        )
        '''
            
        self._init_layers()
        self.init_denoising(dn_cfg)
        self.interaction_acc_all = []
        self.interaction_prec_all = []
        self.interaction_recall_all = []
        self.interaction_f1_all = []
        self.all_pred = []
        self.all_gt = []
        self.no_pair=0

    def _init_layers(self):
        super()._init_layers()
        self.query_embedding = None
        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates [Unknown] class. However, the embedding of unknown class
        # is not used in the original DINO
        self.label_embedding = nn.Embedding(self.cls_out_channels,
                                            self.embed_dims)
        self.downsample = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, self.embed_dims)
        )
        
    def init_denoising(self, dn_cfg):
        if dn_cfg is not None:
            dn_cfg['num_classes'] = self.num_classes
            dn_cfg['num_queries'] = self.num_query
            dn_cfg['hidden_dim'] = self.embed_dims
        self.dn_generator = build_dn_generator(dn_cfg)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      gt_handside=None,
                      gt_interaction=None,
                      **kwargs):
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        assert self.dn_generator is not None, '"dn_cfg" must be set'
        
        if isinstance(gt_handside, DC):
            gt_handside = gt_handside.data  # Already a list of tensors
        if isinstance(gt_interaction, DC):
            gt_interaction = gt_interaction.data  # Already a list of tensors
        '''
        print("[DEBUG forward_train] gt_handside:", gt_handside)
        print("Forward Train Debug:")
        #print("[DEBUG forward_train] keys in data_batch:", data_batch.keys())
        print("[DEBUG train_step] kwargs:", kwargs)

        print("  Batch size:", len(gt_bboxes))
        print("  gt_bboxes:", [b.shape for b in gt_bboxes])
        print("  gt_labels:", [l.shape for l in gt_labels] if gt_labels else None)
        
        print("[DEBUG forward_train] gt_labels:",gt_labels)
        print("[DEBUG forward_train] gt_handside(Before):", gt_handside)
        gt_handside = [x for x in gt_handside]
        print("[DEBUG forward_train] gt_handside (After):", gt_handside)
        # Corrected debug prints for gt_handside (which is now a Tensor)
        
        if gt_handside is not None and gt_handside.numel() > 0:
            print("  gt_handside (in head, tensor):", gt_handside.shape)
            # Depending on if it's batched or not, you might access it differently
            # For a batch of N images, gt_handside might be a single (N, num_objects) tensor
            # Or a list of (num_objects,) tensors.
            # Your current output "tensor([[1, 2, 0, 2]])" suggests it's a single batched tensor.
            print("  First element of gt_handside (in head):", gt_handside[0]) 
        else:
            print("  gt_handside (in head): None or empty tensor.")
        
        '''
        assert len(gt_interaction) == len(gt_handside), f'len(gt_interaction): {len(gt_interaction)} vs len(gt_handside): {len(gt_handside)}'
        #import pdb
        #pdb.set_trace()
        dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
            self.dn_generator(gt_bboxes, gt_labels,
                              self.label_embedding, img_metas)
        #outs = self(x, img_metas, dn_label_query, dn_bbox_query, attn_mask)
        outs, hs = self(x, img_metas, dn_label_query, dn_bbox_query, attn_mask,return_hs=True)
        outputs_classes, outputs_coords, topk_score, topk_anchor, _ = outs
        hs_last = hs[-1]
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas, dn_meta)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, dn_meta)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,gt_handside=gt_handside,hs=hs,gt_interaction=gt_interaction)
        enc_outputs = outs[-1]
        with torch.no_grad():
            tmp_results_list = self.get_bboxes(*outs, img_metas=img_metas, rescale=False, with_nms=False)
            results_list = [res[0] for res in tmp_results_list]
            
        
        #for name, param in self.hand_side_head.named_parameters():
        #    if param.requires_grad:
        #        print(f"[GradCheck] {name}: grad_fn = {param.grad_fn}")

        return losses, enc_outputs, results_list

    def forward(self,
                mlvl_feats,
                img_metas,
                dn_label_query=None,
                dn_bbox_query=None,
                attn_mask=None,
                return_hs=None,
                return_hand_side=None):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        hs, inter_references, topk_score, topk_anchor, enc_outputs = \
            self.transformer(
                mlvl_feats,
                mlvl_masks,
                query_embeds,
                mlvl_positional_encodings,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )
        outs = []
        num_level = len(mlvl_feats)
        start = 0
        for lvl in range(num_level):
            bs, c, h, w = mlvl_feats[lvl].shape
            end = start + h*w
            feat = enc_outputs[start:end].permute(1, 2, 0).contiguous()
            start = end
            outs.append(feat.reshape(bs, c, h, w))
        outs.append(self.downsample(outs[-1]))

        hs = hs.permute(0, 2, 1, 3)

        if dn_label_query is not None and dn_label_query.size(1) == 0:
            # NOTE: If there is no target in the image, the parameters of
            # label_embedding won't be used in producing loss, which raises
            # RuntimeError when using distributed mode.
            hs[0] += self.label_embedding.weight[0, 0] * 0.0

        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        


        if return_hs:
            return (outputs_classes, outputs_coords, topk_score, topk_anchor, outs), hs
        elif return_hand_side:
            hand_logits,_ = self.hand_side_head(hs)  # (6, num_images, num_queries, 2)
            return (outputs_classes, outputs_coords, topk_score, topk_anchor, outs),hand_logits
        else:
            return outputs_classes, outputs_coords, topk_score, topk_anchor, outs
    
    
    
    
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             enc_topk_scores,
             enc_topk_anchors,
             enc_outputs, 
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             dn_meta=None,
             gt_bboxes_ignore=None,
             gt_handside=None,
             hs=None,
             gt_interaction=None):
        # assert gt_bboxes_ignore is None, \
        #     f'{self.__class__.__name__} only supports ' \
        #     f'for gt_bboxes_ignore setting to None.'

        loss_dict = dict()
        import pdb
        #pdb.set_trace()
        # extract denoising and matching part of outputs
        all_cls_scores, all_bbox_preds,hs, dn_cls_scores, dn_bbox_preds, dn_hs = \
            self.extract_dn_outputs(all_cls_scores, all_bbox_preds,hs, dn_meta)
        
        if enc_topk_scores is not None:
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_topk_scores, enc_topk_anchors,
                                 gt_bboxes_list, gt_labels_list,
                                 img_metas, gt_bboxes_ignore)

            # collate loss from encode feature maps
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        # calculate loss from all decoder layers
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_handside_list = [gt_handside for _ in range(num_dec_layers)]
        all_gt_interaction_list = [gt_interaction for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        
        import pdb
        #pdb.set_trace()
        #losses_cls, losses_bbox, losses_iou, losses_hand, hand_acc = multi_apply(
        losses_cls, losses_bbox, losses_iou, loss_interaction,interaction_acc = multi_apply(    
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list, hs,  # from decoder output
            all_gt_handside_list,all_gt_interaction_list)  # per image GT handside)

        # collate loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_interaction'] = loss_interaction[-1]
        
        valid_acc = [a for a in interaction_acc if a is not None]
        loss_dict['interaction_acc'] = torch.stack(valid_acc).mean()

        # collate loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i,loss_interaction_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1],
                                                        loss_interaction[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_interaction'] = loss_interaction_i
            num_dec_layer += 1

        if dn_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_meta = [dn_meta for _ in img_metas]
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                dn_cls_scores, dn_bbox_preds, gt_bboxes_list, gt_labels_list,
                img_metas, dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1],
                    dn_losses_iou[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                num_dec_layer += 1
        

        return loss_dict

    def loss_dn(self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list,
                gt_labels_list, img_metas, dn_meta):
        num_dec_layers = len(dn_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        dn_meta_list = [dn_meta for _ in range(num_dec_layers)]
        return multi_apply(self.loss_dn_single, dn_cls_scores, dn_bbox_preds,
                           all_gt_bboxes_list, all_gt_labels_list,
                           img_metas_list, dn_meta_list)

    def loss_dn_single(self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list,
                       gt_labels_list, img_metas, dn_meta):
        num_imgs = dn_cls_scores.size(0)
        bbox_preds_list = [dn_bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_dn_target(bbox_preds_list, gt_bboxes_list,
                                             gt_labels_list, img_metas,
                                             dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores),
                weight=label_weights,
                avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(  # TODO: How to better return zero loss
                1,
                dtype=cls_scores.dtype,
                device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, dn_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def get_dn_target(self, dn_bbox_preds_list, gt_bboxes_list, gt_labels_list,
                      img_metas, dn_meta):
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_dn_target_single,
                                      dn_bbox_preds_list, gt_bboxes_list,
                                      gt_labels_list, img_metas, dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_dn_target_single(self, dn_bbox_pred, gt_bboxes, gt_labels,
                              img_meta, dn_meta):
        num_groups = dn_meta['num_dn_group']
        pad_size = dn_meta['pad_size']
        assert pad_size % num_groups == 0
        single_pad = pad_size // num_groups
        num_bboxes = dn_bbox_pred.size(0)

        if len(gt_labels) > 0:
            t = torch.range(0, len(gt_labels) - 1).long().cuda()
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = (torch.tensor(range(num_groups)) *
                        single_pad).long().cuda().unsqueeze(1) + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = torch.tensor([]).long().cuda()
        neg_inds = pos_inds + single_pad // 2

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(dn_bbox_pred)
        bbox_weights = torch.zeros_like(dn_bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = dn_bbox_pred.new_tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    @staticmethod
    def extract_dn_outputs(all_cls_scores, all_bbox_preds, hs, dn_meta):
        if dn_meta is not None:
            denoising_cls_scores = all_cls_scores[:, :, :
                                                  dn_meta['pad_size'], :]
            denoising_bbox_preds = all_bbox_preds[:, :, :
                                                  dn_meta['pad_size'], :]
            
            matching_cls_scores = all_cls_scores[:, :, dn_meta['pad_size']:, :]
            matching_bbox_preds = all_bbox_preds[:, :, dn_meta['pad_size']:, :]
            
            if hs is not None:
                denoising_hs = hs[:, :, : dn_meta['pad_size'], :]
                matching_hs = hs[:, :, dn_meta['pad_size']:, :]
            else:
                denoising_hs = None
                matching_hs = None
        else:
            denoising_cls_scores = None
            denoising_bbox_preds = None
            denoising_hs = None
            matching_cls_scores = all_cls_scores
            matching_bbox_preds = all_bbox_preds
            matching_hs = hs
        return (matching_cls_scores, matching_bbox_preds,matching_hs, denoising_cls_scores,
                denoising_bbox_preds,matching_hs)

    def forward_aux(self, mlvl_feats, img_metas, aux_targets, head_idx):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        aux_coords, aux_labels, aux_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks = aux_targets
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        hs, inter_references = self.transformer.forward_aux(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    aux_coords,
                    pos_feats=aux_feats,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                    return_encoder_output=True,
                    attn_masks=attn_masks,
                    head_idx=head_idx
            )

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return outputs_classes, outputs_coords, \
                None, None


    def calculate_metrics(self,pred, gt):
        pred = pred.bool().reshape(-1)
        gt   = gt.bool().reshape(-1)

        tp = (pred & gt).sum().float()
        fp = (pred & ~gt).sum().float()
        fn = (~pred & gt).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        return precision.item(), recall.item(), f1.item()

    def build_pair_geom_norm(self, b1, b2, eps=1e-6):
        # b1, b2: [N,4] normalized (cx,cy,w,h) in [0,1]
        # return: [N, 10] geom features
        cx1, cy1, w1, h1 = b1.unbind(-1)
        cx2, cy2, w2, h2 = b2.unbind(-1)

        diag = (2.0 ** 0.5)
        dx = cx2 - cx1
        dy = cy2 - cy1
        dx_n = dx / (diag + eps)
        dy_n = dy / (diag + eps)

        dist  = torch.sqrt(dx*dx + dy*dy)
        dist_n = dist / (diag + eps)
        cos_t = dx / (dist + eps)
        sin_t = dy / (dist + eps)

        log_w = torch.log((w2 / (w1 + eps)).clamp(min=eps))
        log_h = torch.log((h2 / (h1 + eps)).clamp(min=eps))
        log_a = torch.log(((w2 * h2) / (w1 * h1 + eps)).clamp(min=eps))
        ar1   = torch.log((w1 / (h1 + eps)).clamp(min=eps))
        ar2   = torch.log((w2 / (h2 + eps)).clamp(min=eps))

        # IoU in normalized coords
        x11 = cx1 - w1/2; y11 = cy1 - h1/2
        x12 = cx1 + w1/2; y12 = cy1 + h1/2
        x21 = cx2 - w2/2; y21 = cy2 - h2/2
        x22 = cx2 + w2/2; y22 = cy2 + h2/2
        ix1 = torch.maximum(x11, x21); iy1 = torch.maximum(y11, y21)
        ix2 = torch.minimum(x12, x22); iy2 = torch.minimum(y12, y22)
        iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
        inter  = iw * ih
        area1  = w1 * h1
        area2  = w2 * h2
        iou = inter / (area1 + area2 - inter + eps)

        return torch.stack([dx_n, dy_n, dist_n, cos_t, sin_t,
                            log_w, log_h, log_a, ar1, ar2, iou], dim=-1)

    
    def save_interaction_errors(self,pred, gt_labels, row_idx, col_idx, pos_img_ids,
                                bbox_preds, pos_inds, img_metas, save_root="out/errors"):
        """
        Save FP/FN samples with predicted vs GT interaction errors.

        Args:
            pred: tensor of predicted interaction labels
            gt_labels: tensor of GT interaction labels
            row_idx, col_idx: indices of the paired queries
            pos_img_ids: image IDs for each pos index
            bbox_preds: [bs*Q, 4] normalized cxcywh predicted boxes
            pos_inds: indices of matched queries
            img_metas: list of meta dicts from MMDet
        """
        import os,cv2
        os.makedirs(os.path.join(save_root, "fp"), exist_ok=True)
        os.makedirs(os.path.join(save_root, "fn"), exist_ok=True)

        # --- decode preds back to xyxy (pixel coords of img_shape)
        pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
        pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)  # [Npos, 4]
        
        

        for k in range(len(pred)):
            if pred[k].item() == gt_labels[k].item():
                continue  # skip correct ones

            img_id  = int(pos_img_ids[row_idx[k]])
            meta    = img_metas[img_id]
            path    = meta['filename']
            #if 'AR_AM1gaSvT2E8_11_180_75' in path:
            #    import pdb
            #    pdb.set_trace()
            # load original image and resize to model’s img_shape so coords align
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = meta['img_shape'][:2]
            factor = pos_decode_bbox_pred.new_tensor([W, H, W,H]).unsqueeze(0).repeat(pos_decode_bbox_pred.size(0), 1)
            pos_decode_bbox_pred1 = pos_decode_bbox_pred*factor

            img = cv2.resize(img, (W, H))
            #import pdb
            #pdb.set_trace()
            # get the pair boxes
            box_i = pos_decode_bbox_pred1[row_idx[k]].detach().cpu().numpy().astype(int).tolist()
            box_j = pos_decode_bbox_pred1[col_idx[k]].detach().cpu().numpy().astype(int).tolist()

            # color FP red, FN blue
            is_fp = (pred[k].item() == 1 and gt_labels[k].item() == 0)
            color = (255,0,0) if is_fp else (255,0,0)
            color1 = (0,255,0) if is_fp else (0,255,0)
            tag   = "FP" if is_fp else "FN"

            cv2.rectangle(img, (box_i[0], box_i[1]), (box_i[2], box_i[3]), color1, 2)
            cv2.rectangle(img, (box_j[0], box_j[1]), (box_j[2], box_j[3]), color, 2)
            cv2.putText(img, f"{tag} P:{int(pred[k])} GT:{int(gt_labels[k])}",
                        (max(0, box_i[0]), max(0, box_i[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out_dir = "fp" if is_fp else "fn"
            out_path = os.path.join(save_root, out_dir,
                                    f"{os.path.basename(path).rsplit('.',1)[0]}_{k}.jpg")
            cv2.imwrite(out_path, img)





    def loss_single(self,
                        cls_scores,
                        bbox_preds,
                        gt_bboxes_list,
                        gt_labels_list,
                        img_metas,
                        gt_bboxes_ignore_list=None,
                        hs_tokens=None,                 # NEW: decoder tokens (bs, num_query, embed_dim)
                        gt_handside_list=None,
                        gt_interaction_list=None):         # NEW: list of gt_handside per image
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
            hs_tokens (Tensor, optional): Decoder tokens [bs, num_query, embed_dim].
            gt_handside_list (list[Tensor], optional): Hand side labels per image (0: left, 1: right).

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        import pdb
        #pdb.set_trace()
        #if hs_tokens is not None:
        #    print('hs_tokens: ',hs_tokens.shape)
        if gt_handside_list is not None:
            assert (l.shape == h.shape for l, h in zip(gt_labels_list, gt_handside_list))
            assert (l.shape == h.shape for l, h in zip(gt_labels_list, gt_interaction_list))
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        if gt_handside_list is not None:
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                            gt_bboxes_list, gt_labels_list,
                                            img_metas, gt_bboxes_ignore_list,gt_handside_list,gt_interaction_list)
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg,gt_handside_list_1500,gt_interaction_list_1500) = cls_reg_targets
        else:
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                            gt_bboxes_list, gt_labels_list,
                                            img_metas, gt_bboxes_ignore_list)
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg) = cls_reg_targets   
            gt_handside_list_1500 = None
            gt_interaction_list_1500=None
                    
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        scores = label_weights.new_zeros(labels.shape)
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
        pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
        pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
        scores[pos_inds] = bbox_overlaps(
            pos_decode_bbox_pred.detach(),
            pos_decode_bbox_targets,
            is_aligned=True)
        loss_cls = self.loss_cls(
            cls_scores, (labels, scores),
            weight=label_weights,
            avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0).repeat(
                                            bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        if hs_tokens is not None:
            assert gt_handside_list is not None
            assert gt_handside_list_1500 is not None

            # base det losses already computed above
            loss_interaction = cls_scores.sum() * 0.0
            interaction_acc  = torch.tensor(1.0, device=cls_scores.device)

            # ---- matched queries ----
            matched_tokens = hs_tokens.reshape(-1, hs_tokens.shape[-1])[pos_inds]   # [N_pos, D]
            matched_labels = labels[pos_inds]                                       # [N_pos]; 0=hand,1=first,2=second

            idx_hand   = torch.nonzero(matched_labels == 0, as_tuple=False).squeeze(1)
            idx_first  = torch.nonzero(matched_labels == 1, as_tuple=False).squeeze(1)
            idx_second = torch.nonzero(matched_labels == 2, as_tuple=False).squeeze(1)

            # ---- build raw pairs (HF and optionally FS) ----

            
            
            hf_pairs = (
                torch.cartesian_prod(idx_hand, idx_first)
                if (idx_hand.numel() and idx_first.numel())
                else matched_labels.new_zeros((0, 2), dtype=torch.long)
            )
            #hf_pairs = matched_labels.new_zeros((0, 2), dtype=torch.long)
            
            # keep fs_pairs disabled if you’re checking HF only; otherwise uncomment next block
            
            fs_pairs = (
                 torch.cartesian_prod(idx_first, idx_second)
                 if (idx_first.numel() and idx_second.numel())
                 else matched_labels.new_zeros((0, 2), dtype=torch.long)
            )
            #fs_pairs = matched_labels.new_zeros((0, 2), dtype=torch.long)

            pairs = torch.cat([hf_pairs, fs_pairs], dim=0)  # [P,2]

            # ---- SAME-IMAGE FILTER (restored exactly like before) ----
            row_idx, col_idx = pairs[:, 0], pairs[:, 1]
            Q = hs_tokens.size(1)
            pos_img_ids = (pos_inds // Q)                      # [N_pos]
            same_img = (pos_img_ids[row_idx] == pos_img_ids[col_idx])
            row_idx, col_idx = row_idx[same_img], col_idx[same_img]
            assert row_idx.numel() == 0 or torch.all(pos_img_ids[row_idx] == pos_img_ids[col_idx])

            # ---- GT interaction labels using track ids (no IoU filtering) ----
            matched_interaction = torch.cat(gt_interaction_list_1500, 0).to(cls_scores.device)
            track_ids = [pair[0] for pair in matched_interaction.tolist()]
            idx_to_tid = {i: tid for i, tid in enumerate(track_ids)}

            track_pairs = set()
            for tid1, tid2 in matched_interaction.tolist():
                if tid2 != -1:
                    track_pairs.add((tid1, tid2))
                    track_pairs.add((tid2, tid1))

            P = len(row_idx)
            gt_interaction_labels = torch.zeros(P, dtype=torch.long, device=cls_scores.device)
            for k, (i_local, j_local) in enumerate(zip(row_idx.tolist(), col_idx.tolist())):
                tid_i = idx_to_tid[i_local]
                tid_j = idx_to_tid[j_local]
                gt_interaction_labels[k] = 1 if (tid_i, tid_j) in track_pairs else 0

            # ---- pair features ----
            if P > 0:
                ti = matched_tokens[row_idx]                         # [P, D]
                tj = matched_tokens[col_idx]                         # [P, D]
                token_pairs = torch.cat([ti, tj], dim=-1)            # [P, 2D]

                interaction_logits, loss_interaction = self.interaction_head(
                    token_pairs, gt_interaction_labels
                )
                
                # ======== DO NOT CHANGE ANY OF THE PRINTS BELOW =========
                with torch.no_grad():
                    pred = interaction_logits.argmax(dim=1)
                    correct = (pred == gt_interaction_labels).sum().item()
                    total = gt_interaction_labels.numel()
                    interaction_acc = torch.tensor(
                        (correct / total) if total > 0 else 1.0,
                        device=cls_scores.device
                    )
                    
                #'''    
                    # Precision / Recall / F1
                    tp = ((pred == 1) & (gt_interaction_labels == 1)).sum().item()
                    fp = ((pred == 1) & (gt_interaction_labels == 0)).sum().item()
                    fn = ((pred == 0) & (gt_interaction_labels == 1)).sum().item()

                    prec = tp / (tp + fp + 1e-8)
                    rec  = tp / (tp + fn + 1e-8)
                    f1   = 2 * prec * rec / (prec + rec + 1e-8)

                    prec_t = torch.tensor(prec, device=cls_scores.device)
                    rec_t  = torch.tensor(rec,  device=cls_scores.device)
                    f1_t   = torch.tensor(f1,   device=cls_scores.device)

                    self.interaction_acc_all.append(interaction_acc)
                    self.interaction_prec_all.append(prec_t)
                    self.interaction_recall_all.append(rec_t)
                    self.interaction_f1_all.append(f1_t)

                    if self.interaction_acc_all:
                        if len(self.interaction_acc_all) >= 6:
                            if len(self.interaction_acc_all) % 6 == 0:
                                self.all_pred.append(pred.cpu())
                                self.all_gt.append(gt_interaction_labels.cpu())
                                all_pred = torch.cat(self.all_pred)
                                all_gt   = torch.cat(self.all_gt)
                                acc = (all_pred == all_gt).float().mean().item()
                                tp = ((all_pred == 1) & (all_gt == 1)).sum().item()
                                fp = ((all_pred == 1) & (all_gt == 0)).sum().item()
                                fn = ((all_pred == 0) & (all_gt == 1)).sum().item()
                                prec = tp / (tp + fp + 1e-8)
                                rec  = tp / (tp + fn + 1e-8)
                                f1   = 2 * prec * rec / (prec + rec + 1e-8)
                                print(f"                [FINAL]           Acc: {acc:.3f}, Prec: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, len: {len(all_pred)}, tp: {tp}, fp:{fp}, fn:{fn}")
                                
                                #self.save_interaction_errors(
                                #    pred.cpu(), gt_interaction_labels.cpu(),
                                #    row_idx, col_idx, pos_img_ids, bbox_preds,
                                #    pos_inds, img_metas
                                #)
                                
                            
                            #selected_accuracies_list   = torch.stack(self.interaction_acc_all[5::6]).mean().item()
                            #selected_accuracies_list_p = torch.stack(self.interaction_prec_all[5::6]).mean().item()
                            #selected_accuracies_list_r = torch.stack(self.interaction_recall_all[5::6]).mean().item()
                            #selected_accuracies_list_f = torch.stack(self.interaction_f1_all[5::6]).mean().item()
                            #print(f" ...          last decoder is: Acc: {selected_accuracies_list:.3f}, Prec: {selected_accuracies_list_p:.3f}, Recall: {selected_accuracies_list_r:.3f}, F1: {selected_accuracies_list_f:.3f}, paris: {len(self.interaction_acc_all)}")
                #'''               
                    
                

            else:
                # ---- Dummy loss branch to keep graph identical ----
                C = matched_tokens.size(-1)
                if matched_tokens.size(0) > 0:
                    ti = matched_tokens[0]
                    tj = matched_tokens[0]
                else:
                    ti = torch.ones(C, device=cls_scores.device)
                    tj = ti
                dummy_pair_feat = torch.cat([ti, tj], dim=-1).unsqueeze(0)
                dummy_target    = torch.zeros(1, dtype=torch.long, device=cls_scores.device)
                _, loss_interaction = self.interaction_head(dummy_pair_feat, dummy_target)
                loss_interaction *= 0.0
                interaction_acc = torch.tensor(1.0, device=cls_scores.device)

            return loss_cls, loss_bbox, loss_iou, loss_interaction, interaction_acc

        else:
            return loss_cls, loss_bbox, loss_iou



    def loss_single_aux(self,
                        cls_scores,
                        bbox_preds,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        img_metas,
                        gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        num_q = cls_scores.size(1)
        try:
            labels = labels.reshape(num_imgs * num_q)
            label_weights = label_weights.reshape(num_imgs * num_q)
            bbox_targets = bbox_targets.reshape(num_imgs * num_q, 4)
            bbox_weights = bbox_weights.reshape(num_imgs * num_q, 4)
        except:
            return cls_scores.mean()*0, cls_scores.mean()*0, cls_scores.mean()*0

        bg_class_ind = self.num_classes
        num_total_pos = len(((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1))
        num_total_neg = num_imgs*num_q - num_total_pos

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        scores = label_weights.new_zeros(labels.shape)
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
        pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
        pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
        scores[pos_inds] = bbox_overlaps(
            pos_decode_bbox_pred.detach(),
            pos_decode_bbox_targets,
            is_aligned=True)
        loss_cls = self.loss_cls(
            cls_scores, (labels, scores),
            weight=label_weights,
            avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls*self.lambda_1, loss_bbox*self.lambda_1, loss_iou*self.lambda_1
