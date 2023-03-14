from scipy.optimize import linear_sum_assignment
import torch
from transformers import OwlViTForObjectDetection
from transformers.image_transforms import center_to_corners_format
from torchvision.ops import box_iou
import numpy as np


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

        # Freeze backbone (except layernorms)
        for name, parameter in self.backbone.named_parameters():
            if "layernorm" not in name:
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


class ScaledLoss(torch.nn.Module):
    def __init__(self, class_scales, device, scale_factor=5, cap=250):
        super().__init__()
        self.device = device
        # Class counts don't include the background class (0),
        # which will be set to 1 (no scale)
        max_class = max(class_scales)
        class_scales = [
            min((max_class / count) * scale_factor, cap) for count in class_scales
        ]
        class_scales.insert(0, 1)  # Don't scale background class
        # print(class_scales)
        class_weights = torch.tensor(class_scales, dtype=torch.float).to(self.device)
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction="sum")
        self.cls_loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred_boxes, pred_classes, boxes, labels):
        pred_boxes = pred_boxes.to(self.device)
        pred_classes = pred_classes.to(self.device)
        boxes = boxes.to(self.device)
        labels = labels.to(self.device).float()
        batch_size = pred_boxes.size(0)

        # Index for each batch
        box_loss = torch.tensor(0.0)
        class_loss = torch.tensor(0.0).to(self.device)
        _pred_boxes = []
        for i in range(batch_size):
            _labels = labels[i]
            _boxes = boxes[i]
            _pred_classes = pred_classes[i].to(self.device)

            # Matching
            ious = 1 - box_iou(pred_boxes[i], _boxes)
            lsa, idx = linear_sum_assignment(ious.cpu().detach().numpy())

            # Box loss
            iou_error = (ious[lsa, idx]).sum()
            box_loss = self.smoothl1(pred_boxes[i][lsa], _boxes[idx]) + iou_error
            _pred_boxes.append(pred_boxes[i][lsa].tolist())

            # Class loss
            batch_gts = torch.zeros(len(_pred_classes)).to(self.device)
            batch_gts[lsa] = _labels[idx]
            batch_gts = batch_gts.long()
            class_loss += self.cls_loss(_pred_classes, batch_gts)

        box_loss /= batch_size
        class_loss /= batch_size
        return box_loss, class_loss, torch.tensor(_pred_boxes)
