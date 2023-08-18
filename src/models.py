import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import batched_nms
from transformers import AutoProcessor, OwlViTForObjectDetection
from transformers.image_transforms import center_to_corners_format


# From https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/modules.py
# This is to map the high-dimensional image embedding to an n-dimensional
# space for n-classes
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class OwlViT(torch.nn.Module):
    def __init__(self, pretrained_model, n_classes=81):
        super().__init__()

        # Take the pretrained components that are useful to us
        self.backbone = pretrained_model.owlvit.vision_model
        self.post_layernorm2 = pretrained_model.layer_norm
        self.box_head = pretrained_model.box_head
        self.compute_box_bias = pretrained_model.compute_box_bias
        self.sigmoid = torch.nn.Sigmoid()

        self.class_linear_proj = ProjectionHead(768, n_classes)
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(n_classes, n_classes),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(n_classes, n_classes),
        # )

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

        image_embeddings = self.class_linear_proj(image_embeds)
        # classifier_output = self.classifier(image_embeddings)
        return pred_boxes, image_embeddings, image_embeddings


class PostProcess:
    def __init__(self, mode="cos", confidence_threshold=0.75, iou_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.goalposts = torch.eye(81, dtype=torch.float, device="cuda")
        self.mode = mode

    def __call__(self, all_pred_boxes, class_predictions):
        # Just support batch size of one for now
        pred_boxes = all_pred_boxes.squeeze(0)
        class_predictions.squeeze_(0)

        if self.mode == "ce":
            pred_classes = torch.nn.functional.softmax(class_predictions, dim=-1)
        elif self.mode == "cos":
            class_predictions = torch.nn.functional.normalize(
                class_predictions, p=2, dim=-1
            )
            pred_classes = class_predictions @ self.goalposts
        else:
            # Default to cross entropy
            pred_classes = torch.nn.functional.softmax(class_predictions, dim=-1)

        top = torch.max(pred_classes, dim=1)
        scores = top.values
        classes = top.indices

        idx = (scores > self.confidence_threshold) & (classes != 80)
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

    patched_model = OwlViT(pretrained_model=_model)

    for name, parameter in patched_model.named_parameters():
        conditions = [
            # "layers.9" in name,
            # "layers.10" in name,
            # "layers.11" in name,
            "box" in name,
            "post_layernorm" in name,
            "class_linear_proj" in name,
            # "classifier" in name,
        ]
        if any(conditions):
            continue

        parameter.requires_grad = False

    print("Trainable parameters:")
    for name, parameter in patched_model.named_parameters():
        if parameter.requires_grad:
            print(f"  {name}")
    print()

    # patched_model.load_state_dict(torch.load("epochs/7.pt"), strict=False)
    return patched_model.to(device)
