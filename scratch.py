import json

with open("/mnt/e/datasets/coco/annotations/instances_train2014.json") as f:
    dat = json.load(f)

for d in dat["images"]:
    if d["id"] == 17839:
        print(d)
        break 

for d in dat["annotations"]:
    if d["image_id"] == 17839:
        print(d)
# print(dat["images"][0])
