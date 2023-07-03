import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.functional import softmax
from torchvision.ops import nms
from transformers import AutoProcessor, OwlViTForObjectDetection
from transformers.image_transforms import center_to_corners_format


# Monkey patched for no in-place ops
class PatchedOwlViTClassPredictionHead(nn.Module):
    def __init__(self, original_cls_head):
        super().__init__()

        self.query_dim = original_cls_head.query_dim

        self.dense0 = original_cls_head.dense0
        self.logit_shift = original_cls_head.logit_shift
        self.logit_scale = original_cls_head.logit_scale
        self.elu = original_cls_head.elu

    def forward(
        self, image_embeds, query_embeds, class_masks=None
    ):  # class_masks unused but monkey patch requires 3 args
        image_class_embeds = self.dense0(image_embeds)

        # Normalize image and text features
        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        )
        query_embeds = (
            query_embeds / torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
        )

        # Get class predictions
        pred_logits = torch.einsum(
            "...pd,...qd->...pq", image_class_embeds, query_embeds
        )

        # Apply a learnable shift and scale to logits
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        return (pred_logits, image_class_embeds)


class OwlViT(torch.nn.Module):
    """
    We don't train this that's why it's not an nn.Module subclass.
    We just use this to get to the point where we can use the
    classifier to filter noise.
    """

    def __init__(self, pretrained_model, query_bank):
        super().__init__()

        # Take the pretrained components that are useful to us
        # pretrained_model.class_head = PatchedOwlViTClassPredictionHead(
        #     pretrained_model.class_head
        # )
        self.backbone = pretrained_model.owlvit.vision_model
        self.layernorm = pretrained_model.layer_norm
        self.post_layernorm = pretrained_model.owlvit.vision_model.post_layernorm
        self.class_predictor = PatchedOwlViTClassPredictionHead(
            pretrained_model.class_head
        )
        # self.class_predictor = pretrained_model.class_predictor
        self.box_head = pretrained_model.box_head
        self.compute_box_bias = pretrained_model.compute_box_bias
        self.sigmoid = pretrained_model.sigmoid

        self.queries = torch.nn.Parameter(query_bank, requires_grad=True)

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

    def forward(
        self,
        image: torch.Tensor,
        return_with_embeddings=False,
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
        pred_boxes = self.box_predictor(image_feats, feature_map)

        if return_with_embeddings:
            return pred_boxes, image_feats

        # TODO: monkey patch class head to not use in place ops so I don't have to clone
        pred_class_logits, image_embeds = self.class_predictor(
            image_feats, self.queries
        )

        # TODO: Use these similarities and cosine loss
        # TODO: Add temperature
        # if return_with_logits:
        #     image_embeds = image_embeds / torch.linalg.norm(
        #         image_embeds,
        #         ord=2,
        #         dim=-1,
        #         keepdim=True,
        #     )

        #     queries = queries / torch.linalg.norm(
        #         queries, ord=2, dim=-1, keepdim=True
        #     ).squeeze(0)

        #     image_embeds.squeeze_(0)
        #     queries.squeeze_(0)

        #     logits = torch.matmul(queries, image_embeds.t()).t()

        #     return pred_boxes, pred_classes, logits.unsqueeze(0)

        return pred_boxes, pred_class_logits


class PostProcess:
    def __init__(self, confidence_threshold=0.75, iou_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, all_pred_boxes, pred_classes):
        # Just support batch size of one for now
        pred_boxes = all_pred_boxes.squeeze(0)
        pred_classes = pred_classes.squeeze(0)

        scores = softmax(pred_classes, dim=-1)[:, :-1]
        top = torch.max(scores, dim=1)
        scores = top.values
        classes = top.indices

        idx = scores > self.confidence_threshold

        scores = scores[idx]
        classes = classes[idx]
        pred_boxes = pred_boxes[idx]

        # NMS
        idx = nms(pred_boxes, scores, iou_threshold=self.iou_threshold)
        classes = classes[idx]
        pred_boxes = pred_boxes[idx]
        scores = scores[idx]

        return pred_boxes.unsqueeze_(0), classes.unsqueeze_(0), scores.unsqueeze_(0)


def load_model(labelmap, device, freeze_last_backbone_layer=True):
    _model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    _processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")

    print("Initializing priors from labels...")
    inputs = _processor(
        text=[list(labelmap.values())],
        images=Image.new("RGB", (224, 224)),
        return_tensors="pt",
    )

    with torch.no_grad():
        queries = _model(**inputs).text_embeds

    patched_model = OwlViT(pretrained_model=_model, query_bank=queries)

    for name, parameter in patched_model.backbone.named_parameters():
        # Freeze all layers in vision encoder except last layer
        if not freeze_last_backbone_layer and ".11." in name:
            continue

        parameter.requires_grad = False

    for parameter in patched_model.box_head.parameters():
        parameter.requires_grad = False

    print("\nTrainable parameters:")
    for name, parameter in patched_model.named_parameters():
        if parameter.requires_grad:
            print(name)

    return patched_model.to(device)
