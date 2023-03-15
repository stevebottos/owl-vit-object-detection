import os
import json

import torch
import cv2
from torchvision.ops import box_convert, nms
import numpy as np
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from torch.nn.functional import softmax

from util import scale_bounding_box, draw_box_on_image
from data.dataset import get_dataloaders
from models import OwlViT, FocalBoxLoss

N_EPOCHS = 30
CONFIDENCE_THRESHOLD = 0.75
IOU_THRESHOLD = 0.3

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (
        train_dataloader,
        test_dataloader,
        labelmap,
        test_gts,
    ) = get_dataloaders(
        train_images=2500,
    )

    # Reverse the labelmal for eval
    # reverse_labelmap = {
    #     v["new_idx"]: {"actual_category": k, "name": v["name"]}
    #     for k, v in labelmap.items()
    # }

    model = OwlViT(num_classes=len(labelmap)).to(device)
    criterion = FocalBoxLoss(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(N_EPOCHS):
        os.makedirs(f"debug/{epoch}", exist_ok=True)
        box_losses = []
        cls_losses = []
        for i, (image, labels, boxes, metadata) in enumerate(
            tqdm(train_dataloader, ncols=60)
        ):
            # Some images don't have box annotations. Just skip these
            if boxes.size(1) == 0:
                continue

            image_cv2 = cv2.imread(metadata["impath"].pop())
            model.zero_grad()

            # Prep inputs
            image = image.to(device)
            labels = labels.to(device)

            # coco format is this
            boxes = box_convert(boxes, "xywh", "xyxy")
            boxes = scale_bounding_box(
                boxes, metadata["width"], metadata["height"], mode="down"
            ).to(device)

            pred_boxes, pred_classes = model(image)

            # Predict classes
            box_loss, cls_loss, _pred_boxes = criterion(
                pred_boxes, pred_classes, boxes, labels
            )

            loss = box_loss + cls_loss
            loss.backward()
            optimizer.step()
            box_losses.append(box_loss.item())
            cls_losses.append(cls_loss.item())

            # _pred_boxes = scale_bounding_box(
            #     _pred_boxes, metadata["width"], metadata["height"], mode="up"
            # )

            # for box in _pred_boxes[0].tolist():
            #     image_cv2 = draw_box_on_image(image_cv2, box)
            # cv2.imwrite(f"debug/{epoch}/{i}.jpg", image_cv2)

        if len(box_losses):
            print(round(np.mean(box_losses), 3), "\t", round(np.mean(cls_losses), 3))

    model.eval()
    with torch.no_grad():
        for i, (image, labels, boxes, metadata) in enumerate(
            tqdm(test_dataloader, ncols=60)
        ):
            if i == 1000:
                break

            image_cv2 = cv2.imread(metadata["impath"].pop())

            # Some images don't have box annotations
            if boxes.size(1) == 0:
                continue

            # Prep inputs
            image = image.to(device)
            labels = labels.to(device)

            # coco format to xyxy
            boxes = box_convert(boxes, "xywh", "xyxy")
            boxes = scale_bounding_box(
                boxes, metadata["width"], metadata["height"], mode="down"
            ).to(device)

            pred_boxes, pred_classes = model(image)

            # Just support batch size of one for now
            pred_boxes = pred_boxes.squeeze(0)
            pred_classes = pred_classes.squeeze(0)
            # Get the top scores and apply nms
            scores = softmax(pred_classes, dim=-1)[:, 1:]
            top = torch.max(scores, dim=1)
            scores = top.values
            classes = top.indices

            idx = scores > CONFIDENCE_THRESHOLD

            scores = scores[idx]
            classes = classes[idx]
            pred_boxes = pred_boxes[idx]

            # NMS
            idx = nms(pred_boxes, scores, iou_threshold=IOU_THRESHOLD)
            classes += 1  # We got rid of background, so increment classes by 1
            classes = classes[idx]
            pred_boxes = pred_boxes[idx]
            scores = scores[idx]

            pred_boxes = scale_bounding_box(
                pred_boxes.unsqueeze(0).to("cpu"),
                metadata["width"],
                metadata["height"],
                mode="up",
            )

            classes = classes.tolist()
            pred_boxes = pred_boxes.tolist()
            scores = scores.tolist()

            for box in pred_boxes.pop():
                image_cv2 = draw_box_on_image(image_cv2, box)
            cv2.imwrite(f"eval/{i}.jpg", image_cv2)
