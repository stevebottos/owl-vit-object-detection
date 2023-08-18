import torch
from src.matcher import HungarianMatcher, box_iou, generalized_box_iou

# Hyperparams


class PushPullLoss(torch.nn.Module):
    def __init__(self, n_classes, device, one_to_many_iou_threshold=0.75, margin=0.5):
        super().__init__()

        self.background_label = n_classes
        self.device = device
        self.one_to_many_iou_threshold = one_to_many_iou_threshold
        self.margin = margin

        scales = [1] + [4] * 79 + [0.1]
        self.matcher = HungarianMatcher(n_classes)
        self.goalposts = torch.eye(n_classes + 1, dtype=torch.float, device="cuda")
        self.ce = torch.nn.CrossEntropyLoss(
            reduction="none", weight=None  # torch.tensor(scales, device=device)
        )

    def loss_boxes(self, outputs, targets, indices, idx, num_boxes):
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

    def forward(self, inputs):
        batch_size = len(inputs)
        loss_bbox = torch.tensor(0.0, device=self.device)
        loss_giou = torch.tensor(0.0, device=self.device)
        loss_ce = torch.tensor(0.0, device=self.device)

        _image_embeddings = []
        _target_classes = []

        _goalposts = []
        for inp in inputs:
            (
                image_embeddings,
                class_predictions,
                target_classes,
                predicted_boxes,
                target_boxes,
            ) = inp

            in_preds = {
                "pred_logits": image_embeddings,
                "pred_boxes": predicted_boxes,
            }

            in_targets = [
                {"labels": _labels, "boxes": _boxes}
                for _boxes, _labels in zip(target_boxes, target_classes)
            ]

            target_classes, indices, idx = self.matcher(in_preds, in_targets)

            _loss_bbox, _loss_giou = self.loss_boxes(
                in_preds,
                in_targets,
                indices,
                idx,
                num_boxes=sum(len(t["labels"]) for t in in_targets),
            )

            loss_bbox += _loss_bbox
            loss_giou += _loss_giou

            for box, label in zip(predicted_boxes[0], target_classes[0]):
                if label == self.background_label:
                    continue

                iou, _ = box_iou(box.unsqueeze(0), predicted_boxes.squeeze(0))
                idx = iou > self.one_to_many_iou_threshold
                target_classes[idx] = label.item()

            loss_ce_batch = self.ce(class_predictions.transpose(1, 2), target_classes)
            loss_ce_batch = torch.pow(1 - torch.exp(-loss_ce_batch), 2) * loss_ce_batch
            loss_ce_batch[target_classes == 80] *= 0.1  # alpha
            loss_ce += (
                loss_ce_batch[target_classes == 80].mean()
                + loss_ce_batch[target_classes != 80].mean()
            )

            # CLIP-Style contrastive loss
            image_embeddings = image_embeddings.squeeze(0)
            target_classes = target_classes.squeeze(0)
            for image_embedding, label in zip(image_embeddings, target_classes):
                # if label == self.background_label:
                #     continue
                _image_embeddings.append(image_embedding)
                _target_classes.append(label)
                _goalposts.append(self.goalposts[label])

        # Class loss
        goalposts = torch.stack(_goalposts)
        labels = torch.tensor(
            _target_classes, dtype=torch.float, device="cuda"
        ).unsqueeze(1)
        targets = -torch.clamp(torch.cdist(labels, labels), 0, 1) + 1

        # Class loss from here down
        image_embeddings = torch.stack(_image_embeddings)
        image_embeddings_norm = torch.nn.functional.normalize(
            image_embeddings, p=2, dim=-1
        )

        sims = image_embeddings_norm @ goalposts.t()
        loss_pos = (1 - sims[targets == 1.0]).mean()
        loss_neg = torch.maximum(
            torch.tensor(0.0), sims[targets == 0.0] - self.margin
        ).mean()

        loss = loss_pos + loss_neg
        losses = {
            "loss_ce": loss_ce / batch_size,
            "loss_bg": loss / batch_size,
            "loss_bbox": loss_bbox / batch_size,
            "loss_giou": loss_giou / batch_size,
        }
        return losses, image_embeddings, labels
