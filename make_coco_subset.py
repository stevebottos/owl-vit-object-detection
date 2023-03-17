import json
import random
from collections import Counter

source_annotations_file = "/mnt/e/datasets/coco/annotations/instances_train2014.json"
num_train_samples = 2000
num_test_samples = 250

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

valid_indices = [i["id"] for i in images]
random.shuffle(valid_indices)
train_indices = valid_indices[:num_train_samples]
test_indices = valid_indices[num_train_samples : num_train_samples + num_test_samples]

while True:
    classes = []
    for a in annotations:
        if a["image_id"] in train_indices:
            coco_train["annotations"].append(a)
            classes.append(a["category_id"])
        elif a["image_id"] in test_indices:
            coco_test["annotations"].append(a)
            classes.append(a["category_id"])
    print(json.dumps(Counter(classes), indent=2))
    accept = input("accept? y/n")

    if accept == "y":
        break


for i in images:
    if i["id"] in train_indices:
        coco_train["images"].append(i)
    elif i["id"] in test_indices:
        coco_test["images"].append(i)

with open("data/train.json", "w") as f:
    json.dump(coco_train, f)

with open("data/test.json", "w") as f:
    json.dump(coco_test, f)
