import json
import os
import shutil

import torch
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.io import write_png
from tqdm import tqdm

from src.losses import CombinedLoss
from src.dataset import get_dataloaders
from src.models import PostProcess, load_model
from src.train_util import (
    coco_to_model_input,
    labels_to_classnames,
    model_output_to_image,
    update_metrics,
)
from src.util import BoxUtil, GeneralLossAccumulator, ProgressFormatter


def get_training_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        return data["training"]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device)
    scaler = torch.cuda.amp.GradScaler()
    general_loss = GeneralLossAccumulator()
    progress_summary = ProgressFormatter()

    if os.path.exists("debug"):
        shutil.rmtree("debug")

    training_cfg = get_training_config()
    train_dataloader, test_dataloader, scales, labelmap = get_dataloaders(
        batch_size=training_cfg["batch_size"]
    )

    model = load_model(labelmap, device)
    criterion = CombinedLoss(len(labelmap))

    postprocess = PostProcess(
        confidence_threshold=training_cfg["confidence_threshold"],
        iou_threshold=training_cfg["iou_threshold"],
    )

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=float(training_cfg["learning_rate"]),
    #     weight_decay=training_cfg["weight_decay"],
    # )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
    )

    model.train()
    classMAPs = {v: [] for v in list(labelmap.values())}
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(training_cfg["n_epochs"]):
        if training_cfg["save_eval_images"]:
            os.makedirs(f"debug/{epoch}", exist_ok=True)

        # Train loop
        losses = []
        for i, examples in enumerate(tqdm(train_dataloader, ncols=60)):
            batch_loss = 0
            optimizer.zero_grad()
            for example in examples:
                # Prep inputs
                image = example["image"].to(device).unsqueeze(0)
                labels = example["labels"].to(device).unsqueeze(0)
                boxes = example["boxes"].to(device).unsqueeze(0)
                boxes = coco_to_model_input(boxes, example["meta"]).to(device)

                # Predict
                with torch.cuda.amp.autocast():
                    pred_boxes, pred_classes = model(image)
                    losses = criterion(pred_classes, labels, pred_boxes, boxes)
                    batch_loss += (
                        losses["loss_ce"]
                        + losses["loss_bg"]
                        + losses["loss_bbox"]
                        + losses["loss_giou"]
                    )
                    general_loss.update(losses)

            batch_loss /= training_cfg["batch_size"]
            scaler.scale(batch_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()

        train_metrics = general_loss.get_values()
        general_loss.reset()

        # Eval loop
        model.eval()
        with torch.no_grad():
            for i, examples in enumerate(tqdm(test_dataloader, ncols=60)):
                for example in examples:
                    # Prep inputs
                    image = example["image"].to(device).unsqueeze(0)
                    labels = example["labels"].to(device).unsqueeze(0)
                    boxes = example["boxes"].to(device).unsqueeze(0)
                    metadata = example["meta"]
                    boxes = coco_to_model_input(boxes, metadata).to(device)

                    # Get predictions and save output
                    pred_boxes, pred_classes = model(image)
                    pred_boxes, pred_classes, scores = postprocess(
                        pred_boxes, pred_classes
                    )

                    # Use only the top 200 boxes to stay consistent with benchmarking
                    top = torch.topk(scores, min(200, scores.size(-1)))
                    scores = top.values
                    inds = top.indices.squeeze(0)

                    update_metrics(
                        metric,
                        metadata,
                        pred_boxes[:, inds],
                        pred_classes[:, inds],
                        scores,
                        boxes,
                        labels,
                    )

                    if training_cfg["save_eval_images"]:
                        pred_classes_with_names = labels_to_classnames(
                            pred_classes, labelmap
                        )
                        pred_boxes = model_output_to_image(pred_boxes.cpu(), metadata)
                        image_with_boxes = BoxUtil.draw_box_on_image(
                            metadata["impath"].pop(),
                            pred_boxes,
                            pred_classes_with_names,
                        )

                        write_png(image_with_boxes, f"debug/{epoch}/{i}.jpg")

        print("Computing metrics...")
        val_metrics = metric.compute()
        for i, p in enumerate(val_metrics["map_per_class"].tolist()):
            label = labelmap[str(i)]
            classMAPs[label].append(p)

        with open("class_maps.json", "w") as f:
            json.dump(classMAPs, f)

        metric.reset()
        progress_summary.update(epoch, train_metrics, val_metrics)
        progress_summary.print()
