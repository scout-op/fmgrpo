import os
import copy
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
import numpy as np

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from .grid_mask import GridMask
from .LiftSplatShoot import LiftSplatShootEgo
from .core import seq2nodelist, seq2bznodelist, seq2plbznodelist, av2seq2bznodelist
from .core import EvalSeq2Graph_with_start as EvalSeq2Graph

from .encode_centerline import convert_coeff_coord
from .bz_roadnet_reach_dist_eval import get_geom, get_range
from .lane_diffusion import LaneDiffusion
from .flow_matching import FlowMatchingBEV


@MODELS.register_module()
class SeqGrowGraph(MVXTwoStageDetector):
    """Petr3D. nan for all token except label"""
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 lss_cfg=None,
                 grid_conf=None,
                 bz_grid_conf=None,
                 data_aug_conf=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 vis_cfg=None,
                 freeze_pretrain=True,
                 bev_scale=1.0,
                 epsilon=2,
                 max_box_num=700, #>=660+2
                 init_cfg=None,
                 data_preprocessor=None,front_camera_only=False,vis_dir="original",
                 use_lane_diffusion=False,
                 lane_diffusion_cfg=None,
                 lane_diffusion_stage='inference',
                 use_flow_matching=False,
                 flow_matching_cfg=None,
                 flow_matching_stage='inference',
                 use_grpo_loss=False,
                 grpo_cfg=None,
                 ):
        super(SeqGrowGraph, self).__init__(pts_voxel_layer, pts_middle_encoder,
                                                        pts_fusion_layer, img_backbone, pts_backbone,
                                                        img_neck, pts_neck, pts_bbox_head, img_roi_head,
                                                        img_rpn_head, train_cfg, test_cfg, init_cfg,
                                                        data_preprocessor)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.front_camera_only=front_camera_only
        self.vis_dir=vis_dir
        # data_aug_conf = {
        #     'final_dim': (128, 352),
        #     'H': 900, 'W': 1600,
        # }
        # self.up = Up(512, 256, scale_factor=2)
        # view_transformers = []
        # view_transformers.append(
        #     LiftSplatShoot(grid_conf, data_aug_conf, downsample=16, d_in=256, d_out=256, return_bev=True))
        # self.view_transformers = nn.ModuleList(view_transformers)
        # self.view_transformers = LiftSplatShoot(grid_conf, data_aug_conf, downsample=16, d_in=256, d_out=256, return_bev=True)
        self.view_transformers = LiftSplatShootEgo(grid_conf, data_aug_conf, return_bev=True, **lss_cfg)
        self.downsample = lss_cfg['downsample']
        self.final_dim = data_aug_conf['final_dim']

        self.split_connect=571
        self.split_node=572
        self.start = 574
        self.end = 573
        self.summary_split = 570
        self.split_lines=569
        

      
        
        # self.box_range = 200
        # self.coeff_range = 200
        # self.num_classes=4
        # self.category_start = 200
        # self.connect_start = 250 
        self.coeff_start = 350 
        self.idx_start=250
        self.no_known = 575  # n/a and padding share the same label to be eliminated from loss calculation
        self.num_center_classes = 576 
        # self.noise_connect = 572 
        self.noise_label = 569
        # self.noise_coeff = 570
        self.reward_eps = 1e-6
        
        
        self.vis_cfg = vis_cfg
        self.bev_scale = bev_scale
        self.epsilon = epsilon
        self.max_box_num = max_box_num #!暂时没用到

        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf

        self.dx, bx, nx, self.pc_range, ego_points = get_geom(grid_conf)
        self.bz_dx, bz_bx, bz_nx, self.bz_pc_range = get_range(bz_grid_conf)

        # Lane Diffusion integration
        self.use_lane_diffusion = use_lane_diffusion
        self.use_flow_matching = use_flow_matching
        self.use_grpo_loss = use_grpo_loss
        self.grpo_cfg = copy.deepcopy(grpo_cfg) if grpo_cfg is not None else None
        if self.use_lane_diffusion and self.use_flow_matching:
            raise ValueError('LaneDiffusion and FlowMatching cannot be enabled simultaneously.')

        if use_lane_diffusion:
            # Get BEV dimensions from grid_conf
            bev_h = int((grid_conf['ybound'][1] - grid_conf['ybound'][0]) / grid_conf['ybound'][2])
            bev_w = int((grid_conf['xbound'][1] - grid_conf['xbound'][0]) / grid_conf['xbound'][2])
            
            if lane_diffusion_cfg is None:
                lane_diffusion_cfg = {}
            
            lane_diffusion_cfg.update({
                'bev_channels': lss_cfg.get('d_out', 256),
                'bev_h': bev_h,
                'bev_w': bev_w,
            })
            
            self.lane_diffusion = LaneDiffusion(**lane_diffusion_cfg)
            self.lane_diffusion.set_stage(lane_diffusion_stage)
        else:
            self.lane_diffusion = None

        if use_flow_matching:
            bev_h = int((grid_conf['ybound'][1] - grid_conf['ybound'][0]) / grid_conf['ybound'][2])
            bev_w = int((grid_conf['xbound'][1] - grid_conf['xbound'][0]) / grid_conf['xbound'][2])
            if flow_matching_cfg is None:
                flow_matching_cfg = {}
            flow_matching_cfg = flow_matching_cfg.copy()
            flow_matching_cfg.setdefault('bev_channels', lss_cfg.get('d_out', 256))
            flow_matching_cfg.setdefault('bev_h', bev_h)
            flow_matching_cfg.setdefault('bev_w', bev_w)
            self.flow_matching = FlowMatchingBEV(**flow_matching_cfg)
            self.flow_matching.set_stage(flow_matching_stage)
        else:
            self.flow_matching = None

        if freeze_pretrain:
            self.freeze_pretrain()
    
    def freeze_pretrain(self):
        for m in self.img_backbone.parameters():
            m.requires_grad=False
        for m in self.img_neck.parameters():
            m.requires_grad=False
        for m in self.view_transformers.parameters():
            m.requires_grad=False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def _extract_base_bev(self, img, img_metas):
        """Obtain BEV features before any generative enhancement."""
        img_feats = self.extract_img_feat(img, img_metas)
        largest_feat_shape = img_feats[0].shape[3]
        down_level = int(np.log2(self.downsample // (self.final_dim[0] // largest_feat_shape)))
        bev_feats = self.view_transformers(img_feats[down_level], img_metas)
        return bev_feats

    def extract_feat(self, img, img_metas, gt_centerlines=None):
        """Extract features from images and points."""
        bev_feats = self._extract_base_bev(img, img_metas)

        # Apply LaneDiffusion if enabled
        if self.use_lane_diffusion and self.lane_diffusion is not None:
            stage = self.lane_diffusion.current_stage
            
            if stage == 'stage_i':
                # Train LPIM: need GT centerlines
                if gt_centerlines is not None:
                    bev_feats = self.lane_diffusion(bev_feats, gt_centerlines)
            
            elif stage == 'stage_ii':
                # Train LPDM: returns loss, not features
                # This will be handled in forward_pts_train
                pass
            
            elif stage in ['stage_iii', 'inference']:
                # Use diffusion to enhance features
                bev_feats = self.lane_diffusion(bev_feats)

        # Apply Flow Matching if enabled
        if self.use_flow_matching and self.flow_matching is not None:
            stage = self.flow_matching.current_stage
            if stage == 'stage_i':
                if gt_centerlines is None:
                    raise ValueError('Flow Matching stage_i requires gt_centerlines.')
                bev_feats = self.flow_matching(bev_feats, gt_centerlines)
            elif stage == 'stage_ii':
                # Loss handled in training loop
                pass
            elif stage in ['stage_iii', 'inference']:
                bev_feats = self.flow_matching.enhance(bev_feats)
        
        return bev_feats

    def forward_pts_train(self,
                          bev_feats,
                          gt_lines_sequences,
                          img_metas,
                          num_coeff,summary_subgraphs ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        device = bev_feats[0].device

        input_seqs = []

        max_len = max([len(target) for target in gt_lines_sequences])

        coeff_dim = num_coeff * 2
        

        input_seqs=[]
        for gt_lines_sequence in gt_lines_sequences:
            input_seq= [self.start]+gt_lines_sequence+[self.end]+[self.no_known]*(max_len-len(gt_lines_sequence))
            input_seq=torch.tensor(input_seq, device=device).long()
            input_seqs.append(input_seq.unsqueeze(0))

            
 
        input_seqs = torch.cat(input_seqs , dim=0)  # [8,501]
 
        outputs = self.pts_bbox_head(bev_feats, input_seqs, img_metas)[-1, :, :-1, :]

       
        clause_length = 4 + coeff_dim
        n_control = img_metas[0]['n_control']
        
        
        # =============================================
        # 训练可视化
        
   
        # for bi in range(outputs.shape[0]):
        #     try:
        #         pred_line_seq = outputs[bi]
        #         pred_line_seq = pred_line_seq.argmax(-1)
        #         if self.end in pred_line_seq:
        #             stop_idx = (pred_line_seq == self.end).nonzero(as_tuple=True)[0][0]
        #         else:
        #             stop_idx = len(pred_line_seq)
        #         # if self.summary_split in pred_line_seq:
        #         #     start_idx=(pred_line_seq == self.summary_split).nonzero(as_tuple=True)[0][0]
        #         # else:
        #         #     start_idx=-1
        #         # pred_line_seq = pred_line_seq[start_idx+1:stop_idx]
        #         pred_line_seq = pred_line_seq[:stop_idx]
                
        #         pred_graph = EvalSeq2Graph(img_metas[bi]['token'],pred_line_seq.detach().cpu().numpy().tolist(),front_camera_only=self.front_camera_only,pc_range=self.pc_range,dx=self.dx,bz_pc_range=self.bz_pc_range,bz_dx=self.bz_dx)
        #         pred_graph.visualization([200, 200], os.path.join(self.vis_dir,'train'), 'n', 'n')
        #     except:
        #         import traceback
        #         traceback.print_exc()
        #     break

        # =============================================
        
        outputs = outputs.reshape(-1, self.num_center_classes)  # [602, 2003] last layer
        input_seqs=input_seqs[:,1:]
        input_seqs=input_seqs.flatten()
        gt_seqs_pad=input_seqs[input_seqs!=self.no_known]
        outputs=outputs[input_seqs!=self.no_known]

        losses = self.pts_bbox_head.loss_by_feat_seq(outputs, gt_seqs_pad)

        return losses
    
    def loss(self,
             inputs=None,
             data_samples=None,**kwargs):

        img = inputs['img']
        img_metas = [ds.metainfo for ds in data_samples]

        gt_centerlines = None
        need_gt = False
        if self.use_lane_diffusion and self.lane_diffusion is not None:
            if self.lane_diffusion.current_stage in ['stage_i', 'stage_ii']:
                need_gt = True
        if self.use_flow_matching and self.flow_matching is not None:
            if self.flow_matching.current_stage in ['stage_i', 'stage_ii']:
                need_gt = True
        if need_gt:
            gt_centerlines = self._prepare_gt_centerlines(img_metas)
        
        # Extract BEV features (with or without LaneDiffusion)
        bev_feats = self.extract_feat(img=img, img_metas=img_metas, gt_centerlines=gt_centerlines)
        
        if self.bev_scale != 1.0:
            b, c, h, w = bev_feats.shape
            bev_feats = F.interpolate(bev_feats, (int(h * self.bev_scale), int(w * self.bev_scale)))
        
        losses = dict()
        
        # Handle Stage II (generator training) separately
        if self.use_lane_diffusion and self.lane_diffusion is not None:
            if self.lane_diffusion.current_stage == 'stage_ii':
                raw_bev = self._extract_base_bev(img, img_metas)
                diffusion_loss = self.lane_diffusion.forward_stage_ii(raw_bev, gt_centerlines)
                losses['loss_diffusion'] = diffusion_loss

        if self.use_flow_matching and self.flow_matching is not None:
            if self.flow_matching.current_stage == 'stage_ii':
                raw_bev = self._extract_base_bev(img, img_metas)
                flow_loss = self.flow_matching.forward_stage_ii(raw_bev, gt_centerlines)
                losses['loss_flow'] = flow_loss
        
        # Compute decoder loss
        gt_lines_sequences = [img_meta['centerline_sequence'] for img_meta in img_metas]
        summary_subgraphs=[img_meta['summary_subgraph']  if 'summary_subgraph' in  img_meta else [] for img_meta in img_metas]
    
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        losses_pts = self.forward_pts_train(bev_feats,gt_lines_sequences ,
                                            img_metas, num_coeff,summary_subgraphs )
        losses.update(losses_pts)

        custom_grpo_cfg = kwargs.get('grpo_cfg', None)
        grpo_cfg = custom_grpo_cfg if custom_grpo_cfg is not None else self.grpo_cfg
        if (self.use_grpo_loss and grpo_cfg is not None and self.use_flow_matching
                and self.flow_matching is not None
                and self.flow_matching.current_stage == 'stage_iii'):
            grpo_losses = self.loss_grpo(inputs=inputs,
                                         data_samples=data_samples,
                                         grpo_cfg=grpo_cfg)
            losses.update(grpo_losses)
        return losses
    
    def _prepare_gt_centerlines(self, img_metas):
        """
        Extract and prepare GT centerlines from img_metas
        
        Args:
            img_metas: list of metadata dicts
        
        Returns:
            List of centerline coordinates for each sample
        """
        gt_centerlines = []
        for img_meta in img_metas:
            # Extract centerline coordinates
            # centerline_coord is usually in the format needed
            if 'centerline_coord' in img_meta:
                coords = img_meta['centerline_coord']
                # coords should be [N, 2] for N points
                # We need to group them by lanes - this depends on your data format
                # For now, assume each continuous segment is a lane
                gt_centerlines.append([coords])  # Wrap in list as single lane
            else:
                gt_centerlines.append([])  # Empty if no GT
        
        return gt_centerlines

    def _repeat_img_metas(self, img_metas, repeats):
        repeated = []
        for meta in img_metas:
            for _ in range(repeats):
                repeated.append(copy.deepcopy(meta))
        return repeated

    def _compute_sequence_reward(self, pred_seq, gt_seq, reward_cfg=None):
        reward_cfg = reward_cfg or {}
        eps = reward_cfg.get('eps', self.reward_eps)
        ignore_tokens = {
            self.no_known,
            self.start,
            self.end,
            self.summary_split,
            self.split_connect,
            self.split_node,
            self.split_lines,
        }
        if isinstance(pred_seq, np.ndarray):
            pred_seq = pred_seq.tolist()
        pred_tokens = [int(tok) for tok in pred_seq if int(tok) not in ignore_tokens]
        gt_tokens = [int(tok) for tok in gt_seq if int(tok) not in ignore_tokens]
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        overlap = len(set(pred_tokens) & set(gt_tokens))
        precision = overlap / (len(pred_tokens) + eps)
        recall = overlap / (len(gt_tokens) + eps)
        if precision + recall < eps:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall + eps)
        coverage_bonus = reward_cfg.get('coverage_weight', 0.0) * (overlap / (len(gt_tokens) + eps))
        reward = f1 + coverage_bonus
        reward = max(reward_cfg.get('min_reward', 0.0), min(reward, reward_cfg.get('max_reward', 1.0)))
        return float(reward)

    def loss_grpo(self,
                  inputs=None,
                  data_samples=None,
                  grpo_cfg=None):
        """GRPO-style objective to fine-tune Flow Matching with reward guidance."""
        if not (self.use_flow_matching and self.flow_matching is not None):
            raise RuntimeError('Flow Matching must be enabled before calling loss_grpo.')
        if self.flow_matching.current_stage not in ['stage_iii']:
            raise RuntimeError('loss_grpo is intended for stage_iii training of Flow Matching.')

        grpo_cfg = grpo_cfg or {}
        group_size = grpo_cfg.get('group_size', 4)
        noise_std = grpo_cfg.get('noise_std', 0.1)
        num_steps = grpo_cfg.get('num_steps', self.flow_matching.num_inference_steps)
        adv_weight = grpo_cfg.get('adv_weight', 1.0)
        normalize_adv = grpo_cfg.get('normalize_adv', True)
        kl_weight = grpo_cfg.get('kl_weight', 0.0)
        reward_kwargs = grpo_cfg.get('reward_kwargs', None)

        img = inputs['img']
        img_metas = [ds.metainfo for ds in data_samples]
        base_bev = self._extract_base_bev(img, img_metas)
        b = base_bev.shape[0]

        repeated_bev = base_bev.repeat_interleave(group_size, dim=0)
        repeated_metas = self._repeat_img_metas(img_metas, group_size)

        with torch.no_grad():
            sampled_bev = self.flow_matching.enhance(
                repeated_bev, num_steps=num_steps, noise_std=noise_std)
            line_results = self.simple_test_pts(sampled_bev, repeated_metas)

        rewards = []
        for meta, pred in zip(repeated_metas, line_results):
            gt_seq = meta.get('centerline_sequence', [])
            reward = self._compute_sequence_reward(pred['line_seqs'], gt_seq, reward_kwargs)
            rewards.append(reward)

        reward_tensor = base_bev.new_tensor(rewards).view(b, group_size)
        baseline = reward_tensor.mean(dim=1, keepdim=True)
        best_idx = reward_tensor.argmax(dim=1)
        best_rewards = reward_tensor.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        advantages = best_rewards - baseline.squeeze(1)
        if normalize_adv:
            denom = advantages.std(unbiased=False) + self.reward_eps
            advantages = advantages / denom

        best_indices = (torch.arange(b, device=base_bev.device) * group_size + best_idx).long()
        target_feats = sampled_bev.detach()[best_indices]

        deterministic_bev = self.flow_matching.sample(base_bev, num_steps=num_steps, noise_std=0.0)
        per_sample_loss = F.mse_loss(deterministic_bev, target_feats, reduction='none').mean(dim=[1, 2, 3])
        weight = 1.0 + adv_weight * advantages
        grpo_loss = (per_sample_loss * weight).mean()

        if kl_weight > 0:
            kl_term = F.mse_loss(deterministic_bev, base_bev.detach())
            total_loss = grpo_loss + kl_weight * kl_term
        else:
            kl_term = deterministic_bev.new_tensor(0.0)
            total_loss = grpo_loss

        return {
            'loss_grpo': total_loss,
            'loss_grpo_core': grpo_loss.detach(),
            'loss_grpo_kl': kl_term.detach(),
            'grpo_reward_mean': reward_tensor.mean(),
            'grpo_reward_max': best_rewards.mean(),
        }
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_imgs = batch_inputs_dict['img']
        return self.simple_test(batch_input_metas, batch_input_imgs)

    def simple_test_pts(self, pts_feats, img_metas):
        """Test function of point cloud branch."""
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        clause_length = 4 + num_coeff * 2

        device = pts_feats[0].device
        input_seqs = (torch.ones(pts_feats.shape[0], 1).to(device) * self.start).long()
        outs = self.pts_bbox_head(pts_feats, input_seqs, img_metas)
        output_seqs, values = outs
        line_results = []
        for bi in range(output_seqs.shape[0]):
            pred_line_seq = output_seqs[bi]
            pred_line_seq = pred_line_seq[1:]
            if self.end in pred_line_seq:
                stop_idx = (pred_line_seq == self.end).nonzero(as_tuple=True)[0][0]
            else:
                stop_idx = len(pred_line_seq)
                
            pred_line_seq = pred_line_seq[:stop_idx]
            
     
            line_results.append(dict(
                line_seqs = pred_line_seq.detach().cpu().numpy(),
         
            ))
        return line_results

    def simple_test(self, img_metas, img=None):
        """Test function without augmentaiton."""
        

        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        line_results = self.simple_test_pts(
            bev_feats, img_metas)
        i=0
        for result_dict, line_result, img_meta in zip(bbox_list, line_results, img_metas):
            
            result_dict['line_results'] = line_result
            result_dict['token'] = img_meta['token']
            if i==0:
                try:
                    pred_graph = EvalSeq2Graph(img_meta['token'],line_result["line_seqs"],front_camera_only=self.front_camera_only,pc_range=self.pc_range,dx=self.dx,bz_pc_range=self.bz_pc_range,bz_dx=self.bz_dx)
                    pred_graph.visualization([200, 200], os.path.join(self.vis_dir,'test'), 'n', 'n')
                except:
                    import traceback
                    traceback.print_exc()
            i+=1
                

        return bbox_list

