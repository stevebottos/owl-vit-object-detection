from collections import defaultdict
from typing import Dict

import numpy as np
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
            averaged[k] = v / self.n
        return averaged

    def reset(self):
        self.value = 0


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
                image = draw_bounding_boxes(image, _boxes, width=2)
        else:
            for _boxes, _labels in zip(boxes_batch, labels_batch):
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
