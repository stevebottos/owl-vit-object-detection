import os

import torch
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.io import write_png
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.losses import get_criterion
from src.models import PostProcess, load_model
from src.util import BoxUtil, GeneralLossAccumulator, TensorboardLossAccumulator


def get_training_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        return data["training"]


def coco_to_model_input(boxes, metadata):
    """
    absolute xywh -> relative xyxy
    """
    boxes = BoxUtil.box_convert(boxes, "xywh", "xyxy")
    boxes = BoxUtil.scale_bounding_box(
        boxes, metadata["width"], metadata["height"], mode="down"
    )

    return boxes


def model_output_to_image(boxes, metadata):
    # Model outputs in xyxy normalized coordinates, so scale up
    # before overlaying on image
    boxes = BoxUtil.scale_bounding_box(
        boxes, metadata["width"], metadata["height"], mode="up"
    )

    return boxes


def reverse_labelmap(labelmap):
    return {
        v["new_idx"]: {"actual_category": k, "name": v["name"]}
        for k, v in labelmap.items()
    }


def labels_to_classnames(labels, labelmap):
    return [[labelmap[str(l.item())] for l in labels[0]]]


def update_metrics(metric, metadata, pred_boxes, pred_classes, scores, boxes, labels):
    pred_boxes = BoxUtil.scale_bounding_box(
        pred_boxes.cpu(), metadata["width"], metadata["height"], mode="up"
    )
    boxes = BoxUtil.scale_bounding_box(
        boxes.cpu(), metadata["width"], metadata["height"], mode="up"
    )

    preds = []
    for _pred_boxes, _pred_classes, _scores in zip(pred_boxes, pred_classes, scores):
        preds.append(
            {
                "boxes": _pred_boxes,
                "scores": _scores,
                "labels": _pred_classes,
            }
        )

    targets = []
    for _boxes, _classes in zip(boxes, labels):
        targets.append(
            {
                "boxes": _boxes,
                "labels": _classes,
            }
        )

    metric.update(preds, targets)


if __name__ == "__main__":
    try:
        import shutil

        shutil.rmtree("debug")
        shutil.rmtree("logs")
    except:
        ...

    training_cfg = get_training_config()
    train_dataloader, test_dataloader, scales, labelmap = get_dataloaders()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = get_criterion(num_classes=len(labelmap) - 1).to(device)

    model = load_model(labelmap, device)
    postprocess = PostProcess(confidence_threshold=0.5, iou_threshold=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    metric = MeanAveragePrecision(iou_type="bbox").to(device)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(training_cfg["n_epochs"]):
        if training_cfg["save_debug_images"]:
            os.makedirs(f"debug/{epoch}/eval", exist_ok=True)

        # Train loop
        general_loss = GeneralLossAccumulator()
        tensorboard_finegrained_loss = TensorboardLossAccumulator(log_dir="logs")
        for i, (image, labels, boxes, metadata) in enumerate(
            tqdm(train_dataloader, ncols=60)
        ):
            optimizer.zero_grad()

            # Prep inputs
            image = image.to(device)
            labels = labels.to(device)
            boxes = coco_to_model_input(boxes, metadata).to(device)

            with torch.cuda.amp.autocast():
                # Predict
                all_pred_boxes, pred_classes, similarities = model(image)

                preds = {
                    "pred_logits": pred_classes,  # pred_classes,
                    "pred_boxes": all_pred_boxes,
                }
                gts = [
                    {"labels": _labels, "boxes": _boxes}
                    for _boxes, _labels in zip(boxes, labels)
                ]

                losses, metalosses = criterion(preds, gts)

            loss = losses["loss_ce"] + losses["loss_bbox"] + losses["loss_giou"]
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            # Update accumulators
            general_loss.update(losses)
            tensorboard_finegrained_loss.update(
                labels_to_classnames(labels, labelmap).pop(), metalosses["loss_ce"]
            )

        tensorboard_finegrained_loss.write(epoch)
        print(*general_loss.get_values().items(), sep="\n")
        general_loss.reset()

        # Eval loop
        results = []
        model.eval()
        with torch.no_grad():
            for i, (image, labels, boxes, metadata) in enumerate(
                tqdm(test_dataloader, ncols=60)
            ):
                # Prep inputs
                image = image.to(device)
                labels = labels.to(device)
                boxes = coco_to_model_input(boxes, metadata).to(device)

                # Get predictions and save output
                pred_boxes, pred_classes, scores = postprocess(*model(image)[:-1])
                update_metrics(
                    metric,
                    metadata,
                    pred_boxes,
                    pred_classes,
                    scores,
                    boxes,
                    labels,
                )
                pred_classes_with_names = labels_to_classnames(pred_classes, labelmap)
                if training_cfg["save_debug_images"]:
                    pred_boxes = model_output_to_image(pred_boxes.cpu(), metadata)
                    image_with_boxes = BoxUtil.draw_box_on_image(
                        metadata["impath"].pop(),
                        pred_boxes,
                        pred_classes_with_names,
                    )

                    write_png(image_with_boxes, f"debug/{epoch}/eval/{i}.jpg")

        print("Computing metrics...")
        result = metric.compute()
        print(*result.items(), sep="\n")
