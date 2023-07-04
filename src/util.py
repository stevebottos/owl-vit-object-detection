from collections import defaultdict
from typing import Dict

import numpy as np
from tabulate import tabulate
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.ops import box_convert as _box_convert
from torchvision.utils import draw_bounding_boxes


class TensorboardLossAccumulator:
    def __init__(self, log_dir):
        self.class_losses = defaultdict(list)
        self.writer = SummaryWriter(log_dir)

    def update(self, classes, losses):
        for cls, loss in zip(classes, losses):
            self.class_losses[cls].append(loss)

    def write(self, epoch):
        loss_collector = []
        for k, v in self.class_losses.items():
            val = np.mean(v)
            loss_collector.append(val)
            self.writer.add_scalar(f"loss/{k}", val, epoch)
        self.writer.add_scalar("loss/all", np.mean(loss_collector), epoch)
        self.class_losses = defaultdict(list)
        self.writer.flush()


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
            "box loss": [],
            "map@0.5": [],
            "map (S/M/L)": [],
            "mar (S/M/L)": [],
        }

    def update(self, epoch, train_metrics, val_metrics):
        self.table["epoch"].append(epoch)
        self.table["class loss"].append(train_metrics["loss_ce"])
        self.table["box loss"].append(
            train_metrics["loss_bbox"] + train_metrics["loss_giou"]
        )
        self.table["map@0.5"].append(round(val_metrics["map_50"].item(), 3))

        map_s = round(val_metrics["map_small"].item(), 2)
        map_m = round(val_metrics["map_medium"].item(), 2)
        map_l = round(val_metrics["map_large"].item(), 2)

        self.table["map (S/M/L)"].append(f"{map_s}/{map_m}/{map_l}")

        mar_s = round(val_metrics["mar_small"].item(), 2)
        mar_m = round(val_metrics["mar_medium"].item(), 2)
        mar_l = round(val_metrics["mar_large"].item(), 2)

        self.table["mar (S/M/L)"].append(f"{mar_s}/{mar_m}/{mar_l}")

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