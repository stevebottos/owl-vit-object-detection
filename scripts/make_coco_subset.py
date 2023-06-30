import json
import random
from collections import Counter, OrderedDict
import yaml


def load_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)["data"]
        source_annotations_file = data["annotations_file"]
        num_train_samples = data["num_train_images"]
        num_test_samples = data["num_test_images"]

    return source_annotations_file, num_train_samples, num_test_samples


def load_annotations(source_annotations_file):
    print("Loading data...")
    with open(source_annotations_file) as f:
        dat = json.load(f)

        coco_train = {
            "info": dat["info"],
            "images": [],
            "license": dat["licenses"],
            "annotations": [],
            "categories": dat["categories"],
        }

        coco_test = {
            "info": dat["info"],
            "images": [],
            "license": dat["licenses"],
            "annotations": [],
            "categories": dat["categories"],
        }

        images = dat["images"]
        annotations = dat["annotations"]

    return coco_train, coco_test, images, annotations


def shuffle_indices(valid_indices, num_train_samples, num_test_samples):
    random.shuffle(valid_indices)
    train_indices = valid_indices[:num_train_samples]
    test_indices = valid_indices[
        num_train_samples : num_train_samples + num_test_samples
    ]
    return train_indices, test_indices


if __name__ == "__main__":
    source_annotations_file, num_train_samples, num_test_samples = load_config()
    coco_train, coco_test, images, annotations = load_annotations(
        source_annotations_file
    )

    valid_indices = [i["id"] for i in images]
    train_indices, test_indices = shuffle_indices(
        valid_indices, num_train_samples, num_test_samples
    )

    print("Searching for a valid subset...")
    while True:
        classes = []
        for a in annotations:
            if a["image_id"] in train_indices:
                coco_train["annotations"].append(a)
                classes.append(a["category_id"])
            elif a["image_id"] in test_indices:
                coco_test["annotations"].append(a)
                classes.append(a["category_id"])
        print(json.dumps(OrderedDict(Counter(classes).most_common()), indent=2))
        accept = input("accept? (y/n) >")

        if accept == "y":
            break
        else:
            print("Searching for a valid subset (this might take a few seconds)...")
            train_indices, test_indices = shuffle_indices(
                valid_indices, num_train_samples, num_test_samples
            )

    for i in images:
        if i["id"] in train_indices:
            coco_train["images"].append(i)
        elif i["id"] in test_indices:
            coco_test["images"].append(i)

    with open("data/train.json", "w") as f:
        json.dump(coco_train, f)

    with open("data/test.json", "w") as f:
        json.dump(coco_test, f)
