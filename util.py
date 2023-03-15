# from PIL import Image

import torch
from torchvision.ops import box_convert as _box_convert
import cv2
import numpy as np


class AverageMeter:
    def __init__(self):
        self.value = torch.tensor(0).float().cpu()
        self.n = 0

    def update(self, val):
        self.value += val.detach().cpu()
        self.n += 1

    def get_value(self):
        return (self.value / self.n).item()

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
        image: str or np.ndarray,  # cv2 image
        boxes_batch: torch.tensor,
        color=(0, 255, 0),
    ):
        if isinstance(image, str):
            image = cv2.imread(image)

        boxes_batch = boxes_batch.detach().cpu().numpy()
        for single in boxes_batch:
            for box in single:
                print(box)
                image = cv2.rectangle(
                    image,
                    [int(box[0]), int(box[1])],
                    [int(box[2]), int(box[3])],
                    color,
                    2,
                )
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
