def check_text_guided_shapes():
    import requests
    from PIL import Image
    import torch
    from transformers import AutoProcessor, OwlViTForObjectDetection

    processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = [["a photo of a cat", "a photo of a dog"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)
    print(outputs.class_embeds.shape)
    print(outputs.text_embeds.shape)


def check_image_guided_shapes():
    import requests

    from PIL import Image
    import torch
    from transformers import AutoProcessor, OwlViTForObjectDetection

    processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
    query_image = Image.open(requests.get(query_url, stream=True).raw)
    inputs = processor(images=image, query_images=query_image, return_tensors="pt")
    outputs = model.image_guided_detection(**inputs)
    print(outputs.class_embeds.shape)
    # print(outputs.text_embeds.shape)


check_text_guided_shapes()
check_image_guided_shapes()
