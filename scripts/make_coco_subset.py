import json
import random
from collections import Counter, OrderedDict, defaultdict
from copy import copy

import yaml

# This is because the coco classes have weird numbering
convertor = {
    1: {"new_idx": 0, "name": "person"},
    2: {"new_idx": 1, "name": "bicycle"},
    3: {"new_idx": 2, "name": "car"},
    4: {"new_idx": 3, "name": "motorcycle"},
    5: {"new_idx": 4, "name": "airplane"},
    6: {"new_idx": 5, "name": "bus"},
    7: {"new_idx": 6, "name": "train"},
    8: {"new_idx": 7, "name": "truck"},
    9: {"new_idx": 8, "name": "boat"},
    10: {"new_idx": 9, "name": "traffic light"},
    11: {"new_idx": 10, "name": "fire hydrant"},
    13: {"new_idx": 11, "name": "stop sign"},
    14: {"new_idx": 12, "name": "parking meter"},
    15: {"new_idx": 13, "name": "bench"},
    16: {"new_idx": 14, "name": "bird"},
    17: {"new_idx": 15, "name": "cat"},
    18: {"new_idx": 16, "name": "dog"},
    19: {"new_idx": 17, "name": "horse"},
    20: {"new_idx": 18, "name": "sheep"},
    21: {"new_idx": 19, "name": "cow"},
    22: {"new_idx": 20, "name": "elephant"},
    23: {"new_idx": 21, "name": "bear"},
    24: {"new_idx": 22, "name": "zebra"},
    25: {"new_idx": 23, "name": "giraffe"},
    27: {"new_idx": 24, "name": "backpack"},
    28: {"new_idx": 25, "name": "umbrella"},
    31: {"new_idx": 26, "name": "handbag"},
    32: {"new_idx": 27, "name": "tie"},
    33: {"new_idx": 28, "name": "suitcase"},
    34: {"new_idx": 29, "name": "frisbee"},
    35: {"new_idx": 30, "name": "skis"},
    36: {"new_idx": 31, "name": "snowboard"},
    37: {"new_idx": 32, "name": "sports ball"},
    38: {"new_idx": 33, "name": "kite"},
    39: {"new_idx": 34, "name": "baseball bat"},
    40: {"new_idx": 35, "name": "baseball glove"},
    41: {"new_idx": 36, "name": "skateboard"},
    42: {"new_idx": 37, "name": "surfboard"},
    43: {"new_idx": 38, "name": "tennis racket"},
    44: {"new_idx": 39, "name": "bottle"},
    46: {"new_idx": 40, "name": "wine glass"},
    47: {"new_idx": 41, "name": "cup"},
    48: {"new_idx": 42, "name": "fork"},
    49: {"new_idx": 43, "name": "knife"},
    50: {"new_idx": 44, "name": "spoon"},
    51: {"new_idx": 45, "name": "bowl"},
    52: {"new_idx": 46, "name": "banana"},
    53: {"new_idx": 47, "name": "apple"},
    54: {"new_idx": 48, "name": "sandwich"},
    55: {"new_idx": 49, "name": "orange"},
    56: {"new_idx": 50, "name": "broccoli"},
    57: {"new_idx": 51, "name": "carrot"},
    58: {"new_idx": 52, "name": "hot dog"},
    59: {"new_idx": 53, "name": "pizza"},
    60: {"new_idx": 54, "name": "donut"},
    61: {"new_idx": 55, "name": "cake"},
    62: {"new_idx": 56, "name": "chair"},
    63: {"new_idx": 57, "name": "couch"},
    64: {"new_idx": 58, "name": "potted plant"},
    65: {"new_idx": 59, "name": "bed"},
    67: {"new_idx": 60, "name": "dining table"},
    70: {"new_idx": 61, "name": "toilet"},
    72: {"new_idx": 62, "name": "tv"},
    73: {"new_idx": 63, "name": "laptop"},
    74: {"new_idx": 64, "name": "mouse"},
    75: {"new_idx": 65, "name": "remote"},
    76: {"new_idx": 66, "name": "keyboard"},
    77: {"new_idx": 67, "name": "cell phone"},
    78: {"new_idx": 68, "name": "microwave"},
    79: {"new_idx": 69, "name": "oven"},
    80: {"new_idx": 70, "name": "toaster"},
    81: {"new_idx": 71, "name": "sink"},
    82: {"new_idx": 72, "name": "refrigerator"},
    84: {"new_idx": 73, "name": "book"},
    85: {"new_idx": 74, "name": "clock"},
    86: {"new_idx": 75, "name": "vase"},
    87: {"new_idx": 76, "name": "scissors"},
    88: {"new_idx": 77, "name": "teddy bear"},
    89: {"new_idx": 78, "name": "hair drier"},
    90: {"new_idx": 79, "name": "toothbrush"},
}

new_labelmap = {element["new_idx"]: element["name"] for element in convertor.values()}


def load_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)["data"]
        source_annotations_file = data["annotations_file"]
        num_train_samples = data["num_train_images"]
        num_test_samples = data["num_test_images"]

    return source_annotations_file, num_train_samples, num_test_samples


def shuffle_indices(subset_indices, num_train_samples, num_test_samples):
    random.shuffle(subset_indices)
    train_indices = subset_indices[:num_train_samples]
    test_indices = subset_indices[
        num_train_samples : num_train_samples + num_test_samples
    ]
    return train_indices, test_indices


if __name__ == "__main__":
    source_annotations_file, num_train_samples, num_test_samples = load_config()

    with open(source_annotations_file) as f:
        dat = json.load(f)
        images = dat["images"]
        annotations = dat["annotations"]

    _annotations = defaultdict(list)
    for annotation in annotations:
        _annotations[annotation["image_id"]].append(
            {
                "bbox": annotation["bbox"],
                "label": convertor[annotation["category_id"]]["new_idx"],
            }
        )
    annotations = _annotations

    subset_indices = [i["id"] for i in images]
    train_indices, test_indices = shuffle_indices(
        subset_indices, num_train_samples, num_test_samples
    )
    train_imagemap = {
        element["id"]: element["coco_url"]
        for element in images
        if element["id"] in train_indices
    }

    test_imagemap = {
        element["id"]: element["coco_url"]
        for element in images
        if element["id"] in test_indices
    }

    print("Searching for a valid subset...")
    train = {}
    test = {}
    while True:
        classes = []
        for id, fpath in train_imagemap.items():
            train[fpath] = annotations[id]
            classes.extend([new_labelmap[el["label"]] for el in annotations[id]])

        for id, fpath in test_imagemap.items():
            test[fpath] = annotations[id]
            classes.extend([new_labelmap[el["label"]] for el in annotations[id]])

        print(json.dumps(OrderedDict(Counter(classes).most_common()), indent=2))
        accept = input("accept? (y/n) >")

        if accept == "y":
            break
        else:
            print("Searching for a valid subset (this might take a few seconds)...")
            subset_indices = [i["id"] for i in images]
            train_indices, test_indices = shuffle_indices(
                subset_indices, num_train_samples, num_test_samples
            )
            train_imagemap = {
                element["id"]: element["coco_url"]
                for element in images
                if element["id"] in train_indices
            }

            test_imagemap = {
                element["id"]: element["coco_url"]
                for element in images
                if element["id"] in test_indices
            }

    with open("data/train.json", "w") as f:
        json.dump(train, f)

    with open("data/test.json", "w") as f:
        json.dump(test, f)

    with open("data/labelmap.json", "w") as f:
        json.dump(new_labelmap, f)
