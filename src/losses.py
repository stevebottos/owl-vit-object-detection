import torch
from src.matcher import HungarianMatcher, box_iou, generalized_box_iou

LOGFILE = "log.txt"

with open(LOGFILE, "w") as f:
    f.write("")


class CombinedLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.matcher = HungarianMatcher(n_classes)
        self.n_classes = n_classes
        self.class_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def class_loss(self, outputs, target_classes):
        """
        Custom loss that works off of similarities
        """
        src_logits = outputs["pred_logits"]
        target_classes.squeeze_(0)
        src_logits.squeeze_(0)

        targets = torch.nn.functional.one_hot(
            target_classes, self.n_classes + 1
        ).float()
        idx = targets[:, :-1].sum(dim=1)

        loss = self.class_criterion(src_logits, targets)
        loss = torch.pow(1 - torch.exp(-loss), 2) * loss
        pos_loss = loss[idx == 1].sum(dim=1).mean() * 10
        neg_loss = loss[idx == 0].sum(dim=1).mean() * 10

        return pos_loss, neg_loss

    def loss_boxes(self, outputs, targets, indices, idx, num_boxes):
        """
        (DETR box loss)

        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
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
        target_classes, indices, idx = self.matcher(in_preds, in_targets)

        loss_bbox, loss_giou = self.loss_boxes(
            in_preds,
            in_targets,
            indices,
            idx,
            num_boxes=sum(len(t["labels"]) for t in in_targets),
        )

        for box, label in zip(predicted_boxes[0], target_classes[0]):
            if label == self.n_classes:
                continue

            iou, _ = box_iou(box.unsqueeze(0), predicted_boxes.squeeze(0))
            idx = iou > 0.8
            target_classes[idx] = label.item()

        loss_class, loss_background = self.class_loss(in_preds, target_classes)

        losses = {
            "loss_ce": loss_class,
            "loss_bg": loss_background,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        return losses
