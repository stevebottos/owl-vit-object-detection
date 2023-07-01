import json
import os
from tempfile import NamedTemporaryFile

import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.io import write_png
from tqdm import tqdm

from data.dataset import get_dataloaders
from losses import get_criterion
from models import OwlViT, PostProcess
from util import BoxUtil, GeneralLossAccumulator, TensorboardLossAccumulator


def get_training_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        return data["training"]


def coco_to_model_input(boxes, metadata):
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


def invalid_batch(boxes):
    # Some images don't have box annotations. Just skip these
    return boxes.size(1) == 0


def labels_to_classnames(pred_classes, classmap):
    pred_classes_with_names = []
    for _pcwn in pred_classes:
        pred_classes_with_names.append(
            [classmap[_pred_class.item()]["name"] for _pred_class in _pcwn]
        )

    return pred_classes_with_names


if __name__ == "__main__":
    training_cfg = get_training_config()
    train_dataloader, test_dataloader, train_labelcounts = get_dataloaders()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classmap = reverse_labelmap(train_dataloader.dataset.labelmap)
    labelmap = {
        k: v["name"] for k, v in classmap.items()
    }  # for more generic use later on when I generalize to non-coco stuff, as {idx: classname}
    labelmap.update({len(labelmap): "noise"})
    model = OwlViT(labelmap).to(device)
    # for n, p in model.class_head.named_parameters():
    #     print(n, p.device)

    criterion = get_criterion(num_classes=len(classmap)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    postprocess = PostProcess(confidence_threshold=0.95)

    model.train()
    for epoch in range(training_cfg["n_epochs"]):
        if training_cfg["save_debug_images"]:
            os.makedirs(f"debug/{epoch}/train", exist_ok=True)
            os.makedirs(f"debug/{epoch}/eval", exist_ok=True)

        # Train loop
        general_loss = GeneralLossAccumulator()
        tensorboard_finegrained_loss = TensorboardLossAccumulator(log_dir="logs")
        for i, (image, labels, boxes, metadata) in enumerate(
            tqdm(train_dataloader, ncols=60)
        ):
            if invalid_batch(boxes):
                continue

            model.zero_grad()

            # Prep inputs
            image = image.to(device)
            labels = labels.to(device)
            boxes = coco_to_model_input(boxes, metadata).to(device)

            # Predict
            all_pred_boxes, pred_classes, logits = model(image, return_with_logits=True)

            # TODO: Use pred_classes or logits in loss?
            preds = {"pred_logits": logits, "pred_boxes": all_pred_boxes}
            gts = [
                {"labels": _labels, "boxes": _boxes}
                for _boxes, _labels in zip(boxes, labels)
            ]

            losses, metadata = criterion(preds, gts)

            loss = losses["loss_ce"] + losses["loss_bbox"] + losses["loss_giou"]
            loss.backward()
            optimizer.step()

            # Update accumulators
            general_loss.update(losses)
            tensorboard_finegrained_loss.update(
                labels_to_classnames(labels, classmap).pop(), metadata["loss_ce"]
            )

        tensorboard_finegrained_loss.write(epoch)
        print(*general_loss.get_values().items(), sep="\n")
        general_loss.reset()

    #     # Eval loop
    #     results = []
    #     model.eval()
    #     with torch.no_grad():
    #         for i, (image, labels, boxes, metadata) in enumerate(
    #             tqdm(test_dataloader, ncols=60)
    #         ):
    #             if invalid_batch(boxes):
    #                 continue

    #             # Prep inputs
    #             image = image.to(device)
    #             labels = labels.to(device)
    #             boxes = coco_to_model_input(boxes, metadata).to(device)

    #             # Get predictions and save output
    #             pred_boxes, pred_classes, scores = postprocess(*model(image))
    #             pred_boxes = pred_boxes.cpu()

    #             pred_classes_with_names = labels_to_classnames(pred_classes, classmap)
    #             pred_boxes = model_output_to_image(pred_boxes, metadata)

    #             if training_cfg["save_debug_images"]:
    #                 image_with_boxes = BoxUtil.draw_box_on_image(
    #                     metadata["impath"].pop(), pred_boxes, pred_classes_with_names
    #                 )
    #                 write_png(image_with_boxes, f"debug/{epoch}/eval/{i}.jpg")

    #             # Write in coco format
    #             pred_boxes = BoxUtil.box_convert(pred_boxes, "xyxy", "xywh")

    #             for _pred_boxes, _pred_classes, _scores in zip(
    #                 pred_boxes, pred_classes, scores
    #             ):
    #                 for _pred_box, _pred_class, _score in zip(
    #                     _pred_boxes.tolist(), _pred_classes.tolist(), _scores.tolist()
    #                 ):
    #                     results.append(
    #                         {
    #                             "image_id": metadata["image_id"].item(),
    #                             "category_id": classmap[_pred_class]["actual_category"],
    #                             "bbox": _pred_box,
    #                             "score": _score,
    #                         }
    #                     )

    #     with NamedTemporaryFile(suffix=".json") as tmp:
    #         with open(tmp.name, "w") as f:
    #             json.dump(results, f)

    #         cocoGT = test_dataloader.dataset.coco
    #         cocoDT = cocoGT.loadRes(tmp.name)
    #         coco_eval = COCOeval(cocoGT, cocoDT, "bbox")
    #         coco_eval.evaluate()
    #         coco_eval.accumulate()
    #         coco_eval.summarize()
