# from PIL import Image

import torch
import cv2

# import numpy as np
# import cv2


def scale_bounding_box(bbox: torch.tensor, imwidth: int, imheight: int, mode: str):
    # scale = torch.tensor([imwidth, imheight, imwidth, imheight])
    if mode == "down":
        bbox[:, :, (0, 2)] /= imwidth
        bbox[:, :, (1, 3)] /= imheight
        return bbox
    elif mode == "up":
        bbox[:, :, (0, 2)] *= imwidth
        bbox[:, :, (1, 3)] *= imheight
        return bbox


# def stamp_image(image, colormap):
#     # White out area for the info
#     image[0:250, 0:310] = (0, 0, 0)
#     _y = 30
#     colormap.update({"ground truth": (100, 100, 100)})
#     for label, color in colormap.items():
#         cv2.putText(
#             img=image,
#             text=label,
#             org=(10, _y),
#             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#             fontScale=1,
#             color=color,
#             thickness=3,
#         )
#         _y += 40
#     return image


def draw_box_on_image(image, box, color=(0, 255, 0)):
    image = cv2.rectangle(
        image,
        [int(box[0]), int(box[1])],
        [int(box[2]), int(box[3])],
        color,
        2,
    )
    return image


# def scale_box(box, w, h, mode):
#     if mode == "down":
#         return np.array(box) / [w, h, w, h]
#     elif mode == "up":
#         return np.array(box) * [w, h, w, h]


# def load_sample(image, data, labelmap, processor):
#     w, h = image.size
#     image = processor(images=image, return_tensors="pt")["pixel_values"]

#     _labels = []
#     _boxes = []
#     for label, boxes in data["boxes"].items():

#         if not len(boxes):
#             continue
#         for b in boxes:
#             _labels.append(labelmap[label])
#             _boxes.append(scale_box(b, w, h, mode="down"))

#     return {
#         "image": image,
#         "labels": torch.tensor(_labels),
#         "boxes": torch.tensor(_boxes).float(),
#     }


# def prepare_samples(samples, labelmap, processor):
#     inputs = []
#     for impath, data in samples.items():
#         image = Image.open(impath)
#         inputs.append(load_sample(image, data, labelmap, processor))
#     return inputs
