import json
import os
import shutil

import torch
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.io import write_png
from tqdm import tqdm

from src.losses import PushPullLoss
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
    train_dataloader, test_dataloader, scales, labelmap = get_dataloaders()

    model = load_model(labelmap, device)

    postprocess = PostProcess(
        confidence_threshold=training_cfg["confidence_threshold"],
        iou_threshold=training_cfg["iou_threshold"],
    )

    criterion = PushPullLoss(len(labelmap), device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.25, verbose=True
    )
    model.train()
    classMAPs = {v: [] for v in list(labelmap.values())}
    for epoch in range(training_cfg["n_epochs"]):
        if training_cfg["save_eval_images"]:
            os.makedirs(f"debug/{epoch}", exist_ok=True)

        # Train loop
        losses = []

        for i, examples in enumerate(
            tqdm(train_dataloader, ncols=60)
            # train_dataloader
        ):
            optimizer.zero_grad()
            # Prep inputs
            outputs = []
            for example in examples:
                optimizer.zero_grad()
                image = example["image"].to(device).unsqueeze(0)
                labels = example["labels"].to(device).unsqueeze(0)
                boxes = example["boxes"].to(device).unsqueeze(0)
                metadata = example["metadata"]

                boxes = coco_to_model_input(boxes, metadata)

                # Predict
                pred_boxes, image_embeddings, class_predictions = model(image)
                outputs.append(
                    [image_embeddings, class_predictions, labels, pred_boxes, boxes]
                )
            losses, _, _ = criterion(outputs)

            loss = (
                losses["loss_ce"]
                + losses["loss_bg"]
                + losses["loss_bbox"]
                + losses["loss_giou"]
            )
            loss.backward()
            optimizer.step()
            general_loss.update(losses)

        train_metrics = general_loss.get_values()
        general_loss.reset()
        torch.save(model.state_dict(), f"epochs/{epoch}.pt")

        # Eval loop
        model.eval()
        os.makedirs(f"val_logits/{epoch}", exist_ok=True)
        with torch.no_grad():
            i = 0
            for i, examples in enumerate(tqdm(test_dataloader, ncols=60)):
                outputs = []
                for example in examples:
                    image = example["image"].to(device).unsqueeze(0)
                    labels = example["labels"].to(device).unsqueeze(0)
                    boxes = example["boxes"].to(device).unsqueeze(0)
                    metadata = example["metadata"]
                    # Prep inputs
                    image = image.to(device)
                    labels = labels.to(device)
                    boxes = coco_to_model_input(boxes, metadata).to(device)

                    # Get predictions and save output
                    pred_boxes, image_embeddings, class_predictions = model(image)
                    outputs.append(
                        [image_embeddings, class_predictions, labels, pred_boxes, boxes]
                    )

                losses, image_embeddings, target_classes = criterion(outputs)

                torch.save(image_embeddings, f"val_logits/{epoch}/{i}_image_embeds.pt")
                torch.save(target_classes, f"val_logits/{epoch}/{i}_labels.pt")
                torch.save(class_predictions, f"val_logits/{epoch}/{i}_class_preds.pt")

                pred_boxes, pred_classes, scores = postprocess(
                    pred_boxes, class_predictions
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
        scheduler.step()
