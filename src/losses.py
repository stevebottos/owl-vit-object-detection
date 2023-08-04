import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_area
import numpy as np

LOGFILE = "log.txt"

with open(LOGFILE, "w") as f:
    f.write("")


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
class HungarianMatcher(torch.nn.Module):
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


class PushPullLoss(torch.nn.Module):
    def __init__(self, n_classes, scales):
        super().__init__()
        self.matcher = HungarianMatcher()
        self.class_criterion = torch.nn.BCELoss(reduction="none", weight=scales)

        self.n_classes = n_classes
        self.null_class_margin = 0.2

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def class_loss(self, outputs, targets, indices):
        """
        Custom loss that works off of similarities
        """
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.n_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        assert target_classes.size(0) == 1  # TODO: batches
        target_classes.squeeze_(0)
        src_logits.squeeze_(0)

        targets_one_hot = torch.nn.functional.one_hot(
            target_classes, self.n_classes + 1
        )[:, :-1]

        rows_with_targets = targets_one_hot.sum(dim=-1)
        sims_of_interest = np.round(src_logits[rows_with_targets == 1].tolist(), 1)

        with open(LOGFILE, "a") as f:
            for line in sims_of_interest:
                for element in line:
                    f.write(str(element) + " ")
                f.write(" " + str(max(line)) + "\n")
            f.write("-----------------------------------------------------------\n")

        loss = self.class_criterion(src_logits, targets_one_hot.float())
        target_loss = loss[targets_one_hot == 1]
        background_loss = loss[torch.where((targets_one_hot == 0) & (src_logits > 0.1))]

        target_loss = target_loss.mean()
        background_loss = (
            torch.pow(1 - torch.exp(-background_loss), 3) * background_loss
        ).mean()

        return target_loss, background_loss

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        (DETR box loss)

        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = torch.nn.functional.l1_loss(
            src_boxes, target_boxes, reduction="none"
        )

        metadata = {}

        loss_bbox = loss_bbox.sum() / num_boxes
        metadata["loss_bbox"] = loss_bbox.tolist()

        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        loss_giou = loss_giou.sum() / num_boxes

        return loss_bbox, loss_giou

    def forward(
        self,
        predicted_classes,
        target_classes,
        predicted_boxes,
        target_boxes,
    ):
        # Format to detr style
        in_preds = {
            "pred_logits": predicted_classes,
            "pred_boxes": predicted_boxes,
        }

        in_targets = [
            {"labels": _labels, "boxes": _boxes}
            for _boxes, _labels in zip(target_boxes, target_classes)
        ]

        indices = self.matcher(in_preds, in_targets)
        # Batch index is meaningless since we use a single batch
        # batch index is like [0,0,0,1,1,1,1,1,1,2,2,2,2,3,3] ... etc depending on number of batches
        # TODO: Generalize to more batches
        # target_classes.squeeze_(0)
        # predicted_classes.squeeze_(0)

        loss_class, loss_background = self.class_loss(in_preds, in_targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(
            in_preds,
            in_targets,
            indices,
            num_boxes=sum(len(t["labels"]) for t in in_targets),
        )
        losses = {
            "loss_ce": loss_class,
            "loss_bg": loss_background,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        return losses
