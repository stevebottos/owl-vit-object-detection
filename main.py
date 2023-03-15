import os

import cv2
import torch
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

from util import BoxUtil, AverageMeter
from data.dataset import get_dataloaders
from models import OwlViT, FocalBoxLoss, PostProcess


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


if __name__ == "__main__":
    n_epochs = 5

    (
        train_dataloader,
        test_dataloader,
        labelmap,
        test_gts,
    ) = get_dataloaders(train_images=1000, test_images=1000)
    classmap = reverse_labelmap(labelmap)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    postprocess = PostProcess(confidence_threshold=0.05)
    model = OwlViT(num_classes=len(labelmap)).to(device)
    criterion = FocalBoxLoss(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(n_epochs):
        os.makedirs(f"debug/{epoch}", exist_ok=True)
        t_cls_loss = []
        t_box_loss = []
        cls_loss = AverageMeter()
        box_loss = AverageMeter()
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
            all_pred_boxes, pred_classes = model(image)

            _box_loss, _cls_loss, _ = criterion(
                all_pred_boxes, pred_classes, boxes, labels
            )

            loss = _box_loss + _cls_loss
            loss.backward()
            optimizer.step()

            box_loss.update(_box_loss)
            cls_loss.update(_cls_loss)

            t_box_loss.append(box_loss.item())
            t_class_loss.append(box_loss.item())

        print(box_loss.get_value(), "\t", cls_loss.get_value())
        box_loss.reset()
        cls_loss.reset()

    model.eval()
    with torch.no_grad():
        for i, (image, labels, boxes, metadata) in enumerate(
            tqdm(test_dataloader, ncols=60)
        ):
            if invalid_batch(boxes):
                continue

            # Prep inputs
            image = image.to(device)
            labels = labels.to(device)
            boxes = coco_to_model_input(boxes, metadata).to(device)

            # Get predictions and save output
            pred_boxes = postprocess(*model(image)).cpu()
            pred_boxes = model_output_to_image(pred_boxes, metadata)
            image_with_boxes = BoxUtil.draw_box_on_image(
                metadata["impath"].pop(), pred_boxes
            )
            cv2.imwrite(f"eval/{i}.jpg", image_with_boxes)
