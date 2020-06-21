# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in FsDet.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from fsdet.utils.logger import log_first_n
from collections import OrderedDict
import datetime

import fsdet.utils.comm as comm
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.config import get_cfg, set_global_cfg
from fsdet.data import MetadataCatalog
from fsdet.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from fsdet.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    verify_results,
)

from fsdet.evaluation import DatasetEvaluator, inference_context, print_csv_format

from fsdet.modeling import META_ARCH_REGISTRY
from fsdet.modeling import GeneralizedRCNN, detector_postprocess
from fsdet.modeling import PROPOSAL_GENERATOR_REGISTRY
from fsdet.modeling.proposal_generator.rpn import RPN
from fsdet.modeling.proposal_generator.rpn_outputs import RPNOutputs, find_top_rpn_proposals
from fsdet.layers import cat
from fsdet.modeling.sampling import subsample_labels
from fsdet.modeling.proposal_generator.rpn_outputs import rpn_losses
from fsdet.utils.events import get_event_storage
import numpy as np

from fsdet.modeling import ROI_HEADS_REGISTRY
from fsdet.modeling import StandardROIHeads
from fsdet.modeling.roi_heads.fast_rcnn import FastRCNNOutputs

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_distill(GeneralizedRCNN):

    def forward(self, batched_inputs, is_distill=False):
        """
        For model_s forward.

        Returns:
            logits, deltas and losses of RPN & ROIHeads.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if not self.training:
            return self.inference(batched_inputs, gt_instances, is_distill)

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses, gt_objectness_logits_s, pred_objectness_logits_s, normalizer = \
                self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses, pred_class_logits_gt_s = self.roi_heads(images, features, proposals, gt_instances, is_distill)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, gt_objectness_logits_s, pred_objectness_logits_s, normalizer, pred_class_logits_gt_s

    def inference(self, batched_inputs, gt_instances=None, is_distill=False, detected_instances=None, do_postprocess=True):
        """
        For model_t forward.

        Returns:
            logits, deltas of RPN & ROIHeads.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, pred_objectness_logits_t = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, pred_class_logits_gt_t = self.roi_heads(images, features, proposals, gt_instances, is_distill)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if is_distill: # do distill when infering
            return pred_objectness_logits_t, pred_class_logits_gt_t
        else: # not do distill when infering after training
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results

@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN_distill(RPN):

    def forward(self, images, features, gt_instances=None):
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)
        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = RPNOutputs_distill(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training:
            losses, gt_objectness_logits_s, pred_objectness_logits_s, normalizer = outputs.losses()
            losses = {k: v * self.loss_weight for k, v in losses.items()}
        else:
            pred_objectness_logits_t = cat(
                [
                    # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                    x.permute(0, 2, 3, 1).flatten()
                    for x in pred_objectness_logits
                ],
                dim=0,
            )

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )
            # For RPN-only models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            # For end-to-end models, the RPN proposals are an intermediate state
            # and this sorting is actually not needed. But the cost is negligible.
            inds = [p.objectness_logits.sort(descending=True)[1] for p in proposals]
            proposals = [p[ind] for p, ind in zip(proposals, inds)]
        if self.training:
            return proposals, losses, gt_objectness_logits_s, pred_objectness_logits_s, normalizer
        else:
            return proposals, pred_objectness_logits_t

class RPNOutputs_distill(RPNOutputs):
    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """

        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwriting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            pos_idx, neg_idx = subsample_labels(
                label, self.batch_size_per_image, self.positive_fraction, 0
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()
        """
        gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
            total number of anchors in image i (i.e., len(anchors[i]))
        gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), B),
            where B is the box dimension
        """
        # Collect all objectness labels and delta targets over feature maps and images
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.pred_objectness_logits]
        num_anchors_per_image = sum(num_anchors_per_map)

        # Stack to: (N, num_anchors_per_image)
        gt_objectness_logits = torch.stack(
            [resample(label) for label in gt_objectness_logits], dim=0
        )

        # Log the number of positive/negative anchors per-image that's used in training
        num_pos_anchors = (gt_objectness_logits == 1).sum().item()
        num_neg_anchors = (gt_objectness_logits == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)

        assert gt_objectness_logits.shape[1] == num_anchors_per_image
        # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
        gt_objectness_logits = torch.split(gt_objectness_logits, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_objectness_logits = cat([x.flatten() for x in gt_objectness_logits], dim=0)

        # Stack to: (N, num_anchors_per_image, B)
        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0)
        assert gt_anchor_deltas.shape[1] == num_anchors_per_image
        B = gt_anchor_deltas.shape[2]  # box dimension (4 or 5)

        # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
        gt_anchor_deltas = torch.split(gt_anchor_deltas, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_anchor_deltas = cat([x.reshape(-1, B) for x in gt_anchor_deltas], dim=0)

        # Collect all objectness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        pred_objectness_logits = cat(
            [
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                x.permute(0, 2, 3, 1).flatten()
                for x in self.pred_objectness_logits
            ],
            dim=0,
        )
        pred_anchor_deltas = cat(
            [
                # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
                #          -> (N*Hi*Wi*A, B)
                x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
                for x in self.pred_anchor_deltas
            ],
            dim=0,
        )

        objectness_loss, localization_loss = rpn_losses(
            gt_objectness_logits,
            gt_anchor_deltas,
            pred_objectness_logits,
            pred_anchor_deltas,
            self.smooth_l1_beta,
        )
        normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}
        return losses, gt_objectness_logits, pred_objectness_logits, normalizer

@ROI_HEADS_REGISTRY.register()
class StandardROIHeads_distill(StandardROIHeads):

    def forward(self, images, features, proposals, targets=None, is_distill=False):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses, pred_class_logits_gt_s = self._forward_box(features_list, proposals, targets, is_distill)
            del targets
            return proposals, losses, pred_class_logits_gt_s
        else:
            pred_instances, pred_class_logits_gt_t = self._forward_box(features_list, proposals, targets, is_distill)
            del targets
            return pred_instances, pred_class_logits_gt_t
    
    def _forward_box(self, features, proposals, targets=None, is_distill=False):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features
        if is_distill:
            box_features_gt = self.box_pooler(features, [x.gt_boxes for x in targets])
            box_features_gt = self.box_head(box_features_gt)
            pred_class_logits_gt, _ = self.box_predictor(box_features_gt)
            del box_features_gt

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            assert is_distill
            return outputs.losses(), pred_class_logits_gt
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            if is_distill:
                return pred_instances, pred_class_logits_gt
            else:
                return pred_instances, {}

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def get_pdf(self, x, log=False):
        assert len(x.shape) == 1
        if log:                
            p_1 = F.logsigmoid(x)
            p_2 = p_1 - x # log(1-sigmoid(x)) = logsigmoid(x) - x
            p = cat([p_1.unsqueeze(1), p_2.unsqueeze(1)], dim=1)
        else:
            p_1 = torch.sigmoid(x)
            p_2 = 1 - p_1 # log(1-sigmoid(x)) = logsigmoid(x) - x
            p = cat([p_1.unsqueeze(1), p_2.unsqueeze(1)], dim=1)
        return p

    def forward(self, y_s, y_t, normalizer=1):
        if len(y_s.shape) == 1:
            p_s = self.get_pdf(y_s/self.T, log=True)
            p_t = self.get_pdf(y_t/self.T, log=False)
            loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) * normalizer
        else:
            p_s = F.log_softmax(y_s/self.T, dim=1)
            p_t = F.softmax(y_t/self.T, dim=1)
            loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

class HintLoss(nn.Module):
    """regression loss from hints"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='sum')

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t) / f_s.shape[0]
        return loss

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """
    def __init__(self, cfg, kd_T):
        super().__init__(cfg) # model_s = model, model.train()
        model_t = self.build_model(cfg)# class GRCNN, instantiate
        model_t.eval()
        self.model_t = model_t
        self.check_pointer_t = DetectionCheckpointer(
            self.model_t, save_dir=cfg.OUTPUT_DIR)

        self.criterion_kd_rpn = DistillKL(kd_T)
        # self.criterion_kd_roi_heads = DistillKL(kd_T)# TODO: grid search, LWF
        self.criterion_kd_roi_heads = HintLoss()
        if torch.cuda.is_available():
            self.criterion_kd_rpn.cuda()
            self.criterion_kd_roi_heads.cuda()

    def distill(self, data):
        # normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        loss_origin, gt_objectness_logits_s, pred_objectness_logits_s, normalizer, pred_class_logits_gt_s = self.model(data, is_distill=True)

        with torch.no_grad():
            pred_objectness_logits_t, pred_class_logits_gt_t = self.model_t(data, is_distill=True)
        #TODO: shape of logits for softmax
        valid_masks = gt_objectness_logits_s >= 0
        criterion_kd_rpn = self.criterion_kd_rpn(pred_objectness_logits_s[valid_masks], pred_objectness_logits_t[valid_masks], normalizer)
        criterion_kd_roi_heads = self.criterion_kd_roi_heads(pred_class_logits_gt_s, pred_class_logits_gt_t)
        loss_distill = {
            "loss_distill_rpn": criterion_kd_rpn, 
            "loss_distill_roiheads": criterion_kd_roi_heads
            }

        loss_dict = {}
        loss_dict.update(loss_origin)
        #TODO: alpha = beta = 1, weights of loss_origin & loss_distill
        loss_dict.update(loss_distill)
        return loss_dict

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.distill(data)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (
            self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
                "iteration", -1
            )
            + 1
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN_distill"
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN_distill"
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads_distill"

    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    # init, build model_t, checkpointer_t
    trainer = Trainer(cfg, args.kd_T) 

    # load weights for model_t
    ckpt=args.path_t
    trainer.check_pointer_t._load_model(trainer.check_pointer_t._load_file(ckpt))
    print('load model teacher checkpoint {}'.format(ckpt))

    # # validate teacher AP
    # res = Trainer.test(cfg, trainer.model_t)
    # if comm.is_main_process():
    #     verify_results(cfg, res)
    
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
