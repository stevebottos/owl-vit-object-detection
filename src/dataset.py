import json
import os
from collections import Counter

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import OwlViTProcessor

TRAIN_ANNOTATIONS_FILE = "data/train.json"
TEST_ANNOTATIONS_FILE = "data/test.json"
LABELMAP_FILE = "data/labelmap.json"


def get_images_dir():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)["data"]

        return data["images_path"]


class OwlDataset(Dataset):
    def __init__(self, image_processor, annotations_file):
        self.images_dir = get_images_dir()
        self.image_processor = image_processor

        with open(annotations_file) as f:
            data = json.load(f)
            n_total = len(data)

        self.data = [{k: v} for k, v in data.items() if len(v)]
        print(f"Dropping {n_total - len(self.data)} examples due to no annotations")

    def load_image(self, idx: int) -> Image.Image:
        url = list(self.data[idx].keys()).pop()
        path = os.path.join(self.images_dir, os.path.basename(url))
        image = Image.open(path).convert("RGB")
        return image, path

    def load_target(self, idx: int):
        annotations = list(self.data[idx].values())

        # values results in a nested list
        assert len(annotations) == 1
        annotations = annotations.pop()

        labels = []
        boxes = []
        for annotation in annotations:
            labels.append(annotation["label"])
            boxes.append(annotation["bbox"])

        return labels, boxes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, path = self.load_image(idx)
        labels, boxes = self.load_target(idx)
        w, h = image.size
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
        }
        image = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        return image, torch.tensor(labels), torch.tensor(boxes), metadata


def get_dataloaders(
    train_annotations_file=TRAIN_ANNOTATIONS_FILE,
    test_annotations_file=TEST_ANNOTATIONS_FILE,
):
    image_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    train_dataset = OwlDataset(image_processor, train_annotations_file)
    test_dataset = OwlDataset(image_processor, test_annotations_file)

    with open(LABELMAP_FILE) as f:
        labelmap = json.load(f)

    train_labelcounts = Counter()
    for i in range(len(train_dataset)):
        train_labelcounts.update(train_dataset.load_target(i)[0])

    # scales must be in order
    scales = []
    for i in sorted(list(train_labelcounts.keys())):
        scales.append(train_labelcounts[i])

    scales = np.array(scales)
    scales = (np.round(np.log(scales.max() / scales) + 3, 1)).tolist()

    train_labelcounts = {}
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    return train_dataloader, test_dataloader, scales, labelmap
