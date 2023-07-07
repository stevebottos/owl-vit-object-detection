from collections import defaultdict
from typing import Dict
import time
import numpy as np
from tabulate import tabulate
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.ops import box_convert as _box_convert
from torchvision.utils import draw_bounding_boxes
from datetime import timedelta


class GeneralLossAccumulator:
    def __init__(self):
        self.loss_values = defaultdict(lambda: 0)
        self.n = 0

    def update(self, losses: Dict[str, torch.tensor]):
        for k, v in losses.items():
            self.loss_values[k] += v.item()
        self.n += 1

    def get_values(self):
        averaged = {}
        for k, v in self.loss_values.items():
            averaged[k] = round(v / self.n, 5)
        return averaged

    def reset(self):
        self.value = 0


class ProgressFormatter:
    def __init__(self):
        self.table = {
            "epoch": [],
            "class loss": [],
            "bg loss": [],
            "box loss": [],
            "map": [],
            "map@0.5": [],
            "map (L/M/S)": [],
            "mar (L/M/S)": [],
            "time elapsed": [],
        }
        self.start = time.time()

    def update(self, epoch, train_metrics, val_metrics):
        self.table["epoch"].append(epoch)
        self.table["class loss"].append(train_metrics["loss_ce"])
        self.table["bg loss"].append(train_metrics["loss_bg"])
        self.table["box loss"].append(
            train_metrics["loss_bbox"] + train_metrics["loss_giou"]
        )
        self.table["map"].append(round(val_metrics["map"].item(), 3))
        self.table["map@0.5"].append(round(val_metrics["map_50"].item(), 3))

        map_s = round(val_metrics["map_small"].item(), 2)
        map_m = round(val_metrics["map_medium"].item(), 2)
        map_l = round(val_metrics["map_large"].item(), 2)

        self.table["map (L/M/S)"].append(f"{map_l}/{map_m}/{map_s}")

        mar_s = round(val_metrics["mar_small"].item(), 2)
        mar_m = round(val_metrics["mar_medium"].item(), 2)
        mar_l = round(val_metrics["mar_large"].item(), 2)

        self.table["mar (L/M/S)"].append(f"{mar_l}/{mar_m}/{mar_s}")

        self.table["time elapsed"].append(
            timedelta(seconds=int(time.time() - self.start))
        )

    def print(self):
        print()
        print(tabulate(self.table, headers="keys"))
        print()


class BoxUtil:
    @classmethod
    def scale_bounding_box(
        cls,
        boxes_batch: torch.tensor,  # [M, N, 4, 4]
        imwidth: int,
        imheight: int,
        mode: str,  # up | down
    ):
        if mode == "down":
            boxes_batch[:, :, (0, 2)] /= imwidth
            boxes_batch[:, :, (1, 3)] /= imheight
            return boxes_batch
        elif mode == "up":
            boxes_batch[:, :, (0, 2)] *= imwidth
            boxes_batch[:, :, (1, 3)] *= imheight
            return boxes_batch

    @classmethod
    def draw_box_on_image(
        cls,
        image: str or torch.tensor,  # cv2 image
        boxes_batch: torch.tensor,
        labels_batch: list = None,
        color=(0, 255, 0),
    ):
        if isinstance(image, str):
            image = read_image(image)
        if labels_batch is None:
            for _boxes in boxes_batch:
                if not len(_boxes):
                    continue
                image = draw_bounding_boxes(image, _boxes, width=2)
        else:
            for _boxes, _labels in zip(boxes_batch, labels_batch):
                if not len(_boxes):
                    continue
                image = draw_bounding_boxes(image, _boxes, _labels, width=2)
        return image

    # see https://pytorch.org/vision/main/generated/torchvision.ops.box_convert.html
    @classmethod
    def box_convert(
        cls,
        boxes_batch: torch.tensor,  # [M, N, 4, 4]
        in_format: str,  # [‘xyxy’, ‘xywh’, ‘cxcywh’]
        out_format: str,  # [‘xyxy’, ‘xywh’, ‘cxcywh’]
    ):
        return _box_convert(boxes_batch, in_format, out_format)
