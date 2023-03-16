from scipy.optimize import linear_sum_assignment
import torch
from torch.nn.functional import softmax
from transformers import OwlViTForObjectDetection
from transformers.image_transforms import center_to_corners_format
from torchvision.ops import box_iou, sigmoid_focal_loss, nms
import numpy as np


class FocalBoxLoss(torch.nn.Module):
    def __init__(self, device, post_reduction_bg_scale=1.25):
        super().__init__()
        self.scale = post_reduction_bg_scale
        self.device = device
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction="sum")

    def forward(self, pred_boxes, pred_classes, boxes, labels):
        batch_size = pred_boxes.size(0)
        box_loss = torch.tensor(0.0).to(self.device)
        class_loss = torch.tensor(0.0).to(self.device)
        debug_boxes = []

        for (
            _pred_boxes,
            _pred_classes,
            _boxes,
            _labels,
        ) in zip(pred_boxes, pred_classes, boxes, labels):
            # Matching
            ious = 1 - box_iou(_pred_boxes, _boxes)
            lsa, idx = linear_sum_assignment(ious.detach().cpu().numpy())

            # Box loss
            iou_error = (ious[lsa, idx]).sum()
            box_loss = self.smoothl1(_pred_boxes[lsa], _boxes[idx]) + iou_error
            debug_boxes.append(_pred_boxes[lsa].tolist())

            # Class loss
            n_predictions, n_classes = _pred_classes.size()
            batch_gts = torch.zeros(n_predictions).to(self.device)
            batch_gts = batch_gts.long()
            batch_gts[lsa] = _labels[idx]

            batch_gts_one_hot = torch.nn.functional.one_hot(
                batch_gts, num_classes=n_classes
            ).float()

            _class_loss = sigmoid_focal_loss(
                _pred_classes,
                batch_gts_one_hot,
                reduction="none",
                alpha=0.75,
                gamma=2,  # default
            )

            # Further reduce the impact of the background,
            # extension of focal loss's alpha param
            # but weighted on a per-image basis
            pos_ind = batch_gts != 0
            neg_ind = batch_gts == 0
            reduction = pos_ind.sum() / neg_ind.sum()
            _class_loss[neg_ind] *= reduction * self.scale

            # Amplify the positive classes
            _class_loss[pos_ind]

            class_loss += _class_loss.sum()

        box_loss /= batch_size
        class_loss /= batch_size

        return box_loss, class_loss, torch.tensor(debug_boxes)


class OwlViT(torch.nn.Module):
    """
    We don't train this that's why it's not an nn.Module subclass.
    We just use this to get to the point where we can use the
    classifier to filter noise.
    """

    def __init__(self, num_classes, width=768):
        super().__init__()

        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.backbone = model.owlvit.vision_model
        self.layernorm = model.layer_norm
        self.post_layernorm = model.owlvit.vision_model.post_layernorm

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.post_layernorm.parameters():
            parameter.requires_grad = False

        self.box_head = model.box_head
        self.compute_box_bias = model.compute_box_bias
        self.sigmoid = model.sigmoid
        del model

        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, num_classes),
        )

    # Copied from transformers.models.clip.modeling_owlvit.OwlViTForObjectDetection.box_predictor
    # Removed some comments and docstring to clear up clutter for now
    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pred_boxes = self.box_head(image_feats)
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return center_to_corners_format(pred_boxes)

    # Copied from transformers.models.clip.modeling_owlvit.OwlViTForObjectDetection.image_embedder
    # Removed some comments and docstring to clear up clutter for now
    def image_embedder(self, pixel_values):
        vision_outputs = self.backbone(pixel_values=pixel_values)
        last_hidden_state = vision_outputs.last_hidden_state
        image_embeds = self.post_layernorm(last_hidden_state)

        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layernorm(image_embeds)

        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        return image_embeds

    def forward(self, image: torch.Tensor):
        # Same naming convention as image_guided_detection
        feature_map = self.image_embedder(image)

        new_size = (
            feature_map.shape[0],
            feature_map.shape[1] * feature_map.shape[2],
            feature_map.shape[3],
        )
        image_feats = torch.reshape(feature_map, new_size)
        # Box predictions
        pred_boxes = self.box_predictor(image_feats, feature_map)

        # New class head that works off image_feats instead of feature_map
        pred_classes = self.cls_head(image_feats)

        return pred_boxes, pred_classes


class PostProcess:
    def __init__(self, confidence_threshold=0.75, iou_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, all_pred_boxes, pred_classes):
        # Just support batch size of one for now
        pred_boxes = all_pred_boxes.squeeze(0)
        pred_classes = pred_classes.squeeze(0)

        scores = softmax(pred_classes, dim=-1)[:, 1:]
        top = torch.max(scores, dim=1)
        scores = top.values
        classes = top.indices

        idx = scores > self.confidence_threshold

        scores = scores[idx]
        classes = classes[idx]
        pred_boxes = pred_boxes[idx]

        # NMS
        idx = nms(pred_boxes, scores, iou_threshold=self.confidence_threshold)
        classes += 1  # We got rid of background, so increment classes by 1
        classes = classes[idx]
        pred_boxes = pred_boxes[idx]
        scores = scores[idx]

        return pred_boxes.unsqueeze_(0), classes.unsqueeze_(0), scores.unsqueeze_(0)
