import os

import numpy as np
import torch
from torchvision.ops import batched_nms
from transformers import OwlViTForObjectDetection
from transformers.image_transforms import center_to_corners_format


class OwlViT(torch.nn.Module):
    """
    We don't train this that's why it's not an nn.Module subclass.
    We just use this to get to the point where we can use the
    classifier to filter noise.
    """

    def __init__(self, pretrained_model, n_classes):
        super().__init__()

        # Take the pretrained components that are useful to us
        self.backbone = pretrained_model.owlvit.vision_model
        self.post_post_layernorm = pretrained_model.layer_norm
        self.box_head = pretrained_model.box_head
        self.compute_box_bias = pretrained_model.compute_box_bias
        self.sigmoid = pretrained_model.sigmoid

        # The custom part
        self.class_predictor = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.GELU(),
            torch.nn.Linear(768, 768),
            torch.nn.GELU(),
            torch.nn.Linear(768, n_classes),
        )

    # Copied from transformers.models.clip.modeling_owlvit.OwlViTForObjectDetection.box_predictor
    # Removed some comments and docstring to clear up clutter
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
        vision_outputs = self.backbone(pixel_values=pixel_values).last_hidden_state
        image_embeds = self.backbone.post_layernorm(vision_outputs)

        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.post_post_layernorm(image_embeds)

        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        return image_embeds

    def forward(
        self,
        image: torch.Tensor,
    ):
        # Same naming convention as image_guided_detection
        feature_map = self.image_embedder(image)
        new_size = (
            feature_map.shape[0],
            feature_map.shape[1] * feature_map.shape[2],
            feature_map.shape[3],
        )

        image_feats = torch.reshape(feature_map, new_size)

        # Box predictions
        pred_boxes = self.box_predictor(
            image_feats,
            feature_map,  # Note that the feature map here is just used for its shape and nothing is actually computed from it
        )

        pred_class_logits = self.class_predictor(image_feats)

        return pred_boxes, pred_class_logits


class PostProcess:
    def __init__(self, confidence_threshold=0.75, iou_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, all_pred_boxes, pred_classes):
        # Just support batch size of one for now
        pred_boxes = all_pred_boxes.squeeze(0)
        pred_classes = pred_classes.squeeze(0)
        pred_classes = torch.nn.functional.softmax(pred_classes, dim=1)[:, :-1]

        top = torch.max(pred_classes, dim=1)
        scores = top.values
        classes = top.indices

        idx = scores > self.confidence_threshold
        scores = scores[idx]
        classes = classes[idx]
        pred_boxes = pred_boxes[idx]

        idx = batched_nms(pred_boxes, scores, classes, iou_threshold=self.iou_threshold)
        classes = classes[idx]
        pred_boxes = pred_boxes[idx]
        scores = scores[idx]

        return pred_boxes.unsqueeze_(0), classes.unsqueeze_(0), scores.unsqueeze_(0)


def load_model(labelmap, device):
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model = OwlViT(pretrained_model=model, n_classes=len(labelmap) + 1)

    for name, parameter in model.named_parameters():
        conditions = [
            "layers.11" in name,
            "box" in name,
            "post_layernorm" in name,
            "class_predictor" in name,
            "queries" in name,
        ]
        if any(conditions):
            continue

        parameter.requires_grad = False

    print("Trainable parameters:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"  {name}")
    print()
    return model.to(device)
