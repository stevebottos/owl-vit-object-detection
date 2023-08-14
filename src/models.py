import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.functional import softmax
from torchvision.ops import nms, batched_nms
from transformers import AutoProcessor, OwlViTForObjectDetection
from transformers.image_transforms import center_to_corners_format
import random


# Monkey patched for no in-place ops
# class PatchedOwlViTClassPredictionHead(nn.Module):
#     def __init__(self, original_cls_head):
#         super().__init__()

#         self.query_dim = original_cls_head.query_dim

#         self.dense0 = original_cls_head.dense0
#         self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=3)

#     def forward(self, image_embeds, query_embeds):
#         image_class_embeds = self.dense0(image_embeds)

#         # Normalize image and text features
#         image_class_embeds = image_class_embeds / (
#             torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
#         )
#         query_embeds = (
#             query_embeds / torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
#         )

#         pred_sims = image_class_embeds @ query_embeds.transpose(1, 2)
#         pred_sims = self.pool(pred_sims)

#         return None, pred_sims


class OwlViT(torch.nn.Module):
    """
    We don't train this that's why it's not an nn.Module subclass.
    We just use this to get to the point where we can use the
    classifier to filter noise.
    """

    def __init__(self, pretrained_model, query_bank):
        super().__init__()

        # Take the pretrained components that are useful to us
        self.backbone = pretrained_model.owlvit.vision_model
        self.post_layernorm2 = pretrained_model.layer_norm
        self.box_head = pretrained_model.box_head
        self.compute_box_bias = pretrained_model.compute_box_bias
        self.sigmoid = torch.nn.Sigmoid()

        # This is our text section
        self.text_layernorm = torch.nn.LayerNorm(512)
        self.queries = torch.nn.Parameter(query_bank)

        # This is our image section
        self.image_layernorm = torch.nn.LayerNorm(512)
        self.class_linear_proj = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, 512),
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

        # Layernorm before melding in the class token
        image_embeds = self.backbone.post_layernorm(last_hidden_state)

        # Add the class token in
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)
        image_embeds = image_embeds[:, 1:, :] * class_token_out

        # Layernorm after class token is mixed in
        image_embeds = self.post_layernorm2(image_embeds)

        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )

        return image_embeds, image_embeds.reshape(new_size)

    def forward(
        self,
        image: torch.Tensor,
    ):
        # Same naming convention as image_guided_detection
        image_embeds, image_embeds_as_feature_map = self.image_embedder(image)

        # Box predictions
        pred_boxes = self.box_predictor(
            image_embeds,
            image_embeds_as_feature_map,  # Not actually used for anything except its shape
        )

        image_embeddings = self.image_layernorm(self.class_linear_proj(image_embeds))
        text_embeddings = self.text_layernorm(self.queries)

        return pred_boxes, image_embeddings, text_embeddings


class PostProcess:
    def __init__(self, confidence_threshold=0.75, iou_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, all_pred_boxes, image_embeddings, text_embeddings):
        # Just support batch size of one for now
        pred_boxes = all_pred_boxes.squeeze(0)
        image_embeddings.squeeze_(0)
        text_embeddings.squeeze_(0)

        ie = (
            image_embeddings / torch.linalg.norm(image_embeddings, dim=-1, keepdim=True)
            + 1e-6
        )

        te = (
            text_embeddings / torch.linalg.norm(text_embeddings, dim=-1, keepdim=True)
            + 1e-6
        )
        pred_classes = ie @ te.T

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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    _model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    _processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")

    print("Initializing priors from labels...")
    labels = list(labelmap.values())
    labels.append("background")
    inputs = _processor(
        text=[labels],
        images=Image.new("RGB", (224, 224)),
        return_tensors="pt",
    )

    with torch.no_grad():
        queries = _model(**inputs).text_embeds
    queries = torch.randn(queries.shape)

    patched_model = OwlViT(pretrained_model=_model, query_bank=queries)

    for name, parameter in patched_model.named_parameters():
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
    for name, parameter in patched_model.named_parameters():
        if parameter.requires_grad:
            print(f"  {name}")
    print()
    return patched_model.to(device)
