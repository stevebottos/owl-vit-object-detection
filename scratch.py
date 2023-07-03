import requests
from PIL import Image
import torch
from transformers import AutoProcessor, OwlViTForObjectDetection, OwlViTModel

processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
model2 = OwlViTModel.from_pretrained("google/owlvit-base-patch32")


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model2(**inputs, return_loss=True)
print(outputs.loss)
# Target image sizes (height, width) to rescale box predictions [batch_size, 2]


# i = 0  # Retrieve predictions for the first image for the corresponding text queries
# text = texts[i]
# boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# for box, score, label in zip(boxes, scores, labels):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#         f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
#     )

# from PIL import Image
# import requests
# from transformers import AutoProcessor, OwlViTModel

# model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
# processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(
#     images=image,
#     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]],
#     return_tensors="pt",
# )
# image_features = model(**inputs, return_loss=True)
# print(image_features.loss)
