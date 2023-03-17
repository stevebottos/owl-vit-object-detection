from scipy.optimize import linear_sum_assignment
import torch
from torch.nn.functional import softmax
from transformers import OwlViTForObjectDetection
from transformers.image_transforms import center_to_corners_format
from torchvision.ops import box_area, nms
import numpy as np
import torch.nn as nn


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# from https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


# From https://github.com/facebookresearch/detr/blob/main/models/matcher.py
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class FocalBoxLoss(torch.nn.Module):
    def __init__(self, device, train_labelcouts, scale=3.0):
        super().__init__()
        max_class = max(train_labelcouts)
        scales = [
            (2 - (classcount / max_class)) * scale for classcount in train_labelcouts
        ]
        scales.insert(0, 1)
        self.scale = scale
        self.device = device
        self.matcher = HungarianMatcher()
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction="sum")
        self.cls_loss = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(scales), reduction="none"
        ).to(self.device)

    def forward(self, pred_boxes, pred_classes, boxes, labels):
        batch_size = pred_boxes.size(0)

        class_loss = torch.tensor(0.0).to(self.device)

        # Prepare inputs for detr matcher
        preds = {"pred_logits": pred_classes, "pred_boxes": pred_boxes}
        gts = [
            {"labels": _labels, "boxes": _boxes}
            for _boxes, _labels in zip(boxes, labels)
        ]

        box_loss = torch.tensor(0.0).to(self.device)
        batch_matches = self.matcher(preds, gts)
        for i, (pred_i, gt_i) in enumerate(batch_matches):
            # box loss
            _pred_boxes = pred_boxes[i][pred_i]
            _gt_boxes = boxes[i][gt_i]
            box_loss += self.smoothl1(_pred_boxes, _gt_boxes)

            # class loss
            _pred_classes = pred_classes[i]
            _labels = labels[i]

            n_predictions, _ = _pred_classes.size()
            batch_gts = torch.zeros(n_predictions).to(self.device).long()
            batch_gts[pred_i] = _labels[gt_i]

            _class_loss = self.cls_loss(
                _pred_classes.to(self.device), batch_gts.to(self.device)
            )

            pos_ind = batch_gts != 0
            pos_loss = _class_loss[pos_ind]
            _class_loss[batch_gts != 0] = 0
            neg_loss, _ = torch.sort(_class_loss, descending=True)
            neg_loss = neg_loss[: pos_ind.sum() * 3]

            # # Apply negative/positive
            # pos_ind = batch_gts != 0
            # neg_ind = batch_gts == 0

            # reduction = pos_ind.sum() / neg_ind.sum()
            # _class_loss[neg_ind] *= reduction
            # _class_loss[pos_ind] *= self.scale

            class_loss += (pos_loss.sum() + neg_loss.sum()) / pos_ind.sum()

        box_loss /= batch_size
        class_loss /= batch_size
        return box_loss, class_loss


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
