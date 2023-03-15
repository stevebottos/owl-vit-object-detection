from collections import OrderedDict, Counter

import os

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import OwlViTProcessor


class OrderedCounter(Counter, OrderedDict):
    pass


class CocoSubset(Dataset):
    def __init__(self, image_processor):
        annotations_file = "/mnt/e/datasets/coco/annotations/instances_train2014.json"
        self.images_dir = "/mnt/e/datasets/coco/train2014"

        self.coco = COCO(annotations_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.labelmap = {
            k: {"new_idx": i + 1, "name": v["name"]}
            for i, (k, v) in enumerate(self.coco.cats.items())
        }

        # 0 is reserved for background
        assert 0 not in self.labelmap.keys()
        self.labelmap[0] = {"new_idx": 0, "name": "background"}
        self.image_processor = image_processor

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = Image.open(os.path.join(self.images_dir, path)).convert("RGB")
        return image, path

    def _load_target(self, id: int):
        annotations = self.coco.loadAnns(self.coco.getAnnIds(id))

        labels = []
        boxes = []
        for annotation in annotations:
            labels.append(self.labelmap[annotation["category_id"]]["new_idx"])
            boxes.append(annotation["bbox"])

        return labels, boxes

    def load_annotation(self, idx):
        id = self.ids[idx]
        labels, _ = self._load_target(id)
        return labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        image, path = self._load_image(id)
        labels, boxes = self._load_target(id)
        w, h = image.size
        metadata = {
            "width": w,
            "height": h,
            "impath": f"{self.images_dir}/{path}",
            "image_id": id,
        }
        image = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        return image, torch.tensor(labels), torch.tensor(boxes), metadata


def get_dataloaders(train_images):
    image_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    dataset = CocoSubset(image_processor)

    train_indices = list(range(train_images))
    test_indices = list(range(train_images, len(dataset)))

    print(
        f"Using indices {min(train_indices)} to {max(train_indices)} as train, indices {min(test_indices)} to {max(test_indices)} as test"
    )

    labelmap = dataset.labelmap
    train_subset = Subset(dataset, indices=train_indices)
    test_subset = Subset(dataset, indices=test_indices)

    # For scaling
    labelcounts = OrderedCounter()
    for i in range(len(dataset)):
        labelcounts.update(train_subset.dataset.load_annotation(i))

    train_dataloader = DataLoader(
        train_subset, batch_size=1, shuffle=False, num_workers=1
    )
    test_dataloader = DataLoader(
        test_subset, batch_size=1, shuffle=False, num_workers=1
    )
    return train_dataloader, test_dataloader, labelmap, dataset.coco
