import torch
import clip
from PIL import Image
import os
import numpy as np


FILE_DIRECTORY = r"D:\Development\Python\PicOrginizer\pics"
RECURSIVE = True

image_extensions =[".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

file_promt = {
    "Selfie": [
        "a close-up selfie photo of a person",
        "a portrait taken with a front camera",
        "a person holding a phone taking a selfie"
    ],
    "Nature": [
        "a landscape photo of nature",
        "a forest, mountain or outdoor scenery",
        "a natural environment with trees, sky or water"
    ],
    "Food": [
        "a photo of food on a plate",
        "a close-up picture of a meal",
        "a dish served on a table"
    ],
    "ScreenShots": [
        "a screenshot of a computer screen",
        "a screenshot of a mobile phone display",
        "a captured screen image with UI elements"
    ],
    "Document": [
        "a photo of a document with text",
        "a scanned paper with printed text",
        "a page of a book or printed document"
    ]
}
file = list(file_promt.keys())
promt = [item for sublist in file_promt.values() for item in sublist]

print(promt)
print(file)

newfile = []
for file in file:
    newfile.append(os.path.join(FILE_DIRECTORY, file))

print(newfile)


path_list = []
output_names = []
output_file_names = []

if RECURSIVE:
    walker = os.walk(FILE_DIRECTORY)
else:
    walker = [(FILE_DIRECTORY, [], os.listdir(FILE_DIRECTORY))]

for root, _, files in walker:
    for file in files:
        full_path = os.path.join(root, file)
        if os.path.isfile(full_path):
            name, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                path_list.append(full_path)
                output_names.append(name)

print(path_list)
print(output_names)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

for path in path_list:
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    text = clip.tokenize(promt).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    max_prob = max(probs[0])
    max_index = np.argmax(probs[0])
    print(f"the {path} Label probs:{probs} Max prob:{max_prob} Max index:{promt[max_index]}")  




