# changes: use open_clip
import open_clip
# keep other imports
import os
import shutil
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm  # Added for progress bar

# --- CONFIG ---
FILE_DIRECTORY = r"E:\PICTURE\Family"
RECURSIVE = True
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

file_promt = {
    "Selfie": [
        "a selfie photo of exactly one person, face close to camera",
        "a single person holding a smartphone taking their own photo",
        "front camera portrait of one individual, upper body visible",
        "mirror selfie showing one person and their phone",
        "close-up face photo, only one person in frame",
        "solo portrait photo, no other people visible",
    ],
    "People_Group": [
        "a group photo with multiple people standing together",
        "two or more people posing for a picture",
        "family or friends group portrait with several faces visible",
        "crowd of people at an event or party",
        "team photo with many people in the frame",
        "multiple people interacting together in one photo",
    ],
    "Nature_Landscape": [
        "wide landscape photo of mountains, forest, sea or fields, scenic travel shot",
        "outdoor nature scenery with sky, horizon, and vegetation",
        "sunset or sunrise over natural landscape, no buildings",
        "panoramic or wide-angle nature photo, daylight",
        "natural scenery photo emphasizing landforms and vegetation",
    ],
    "City_Urban": [
        "urban street photo with buildings, cars and sidewalks",
        "city skyline or downtown architecture, daytime or night lights",
        "street photography showing people and vehicles in a city",
        "urban scene with skyscrapers or busy intersections",
        "night cityscape with illuminated windows and streetlights",
    ],
    "Food": [
        "close-up food photography on a plate, styled dish, restaurant quality",
        "overhead photo of a meal on a table, food blog style",
        "dish served with table setting or utensils visible",
        "dessert or plated meal photo with clear food details",
        "casual home-cooked meal photo on a plate",
    ],
    "Pet_Animals": [
        "photo of a pet dog or cat, animal facing camera, indoor or outdoor",
        "small pet portrait with visible fur and eyes",
        "animal photo with human interaction or toy",
        "wild animal in nature, not a pet (different composition)",
    ],
    "Screenshots": [
        "screenshot of a computer or phone interface with sharp text and icons",
        "mobile app screen capture, UI elements and menu visible",
        "webpage screenshot showing browser chrome and text",
        "terminal or code editor screenshot with monospace text",
        "screenshot containing crisp rectangles, buttons, and readable UI text",
    ],
    "Document_Text": [
        "scanned document or photographed paper with dense printed text",
        "official form, invoice, or typed page on a desk",
        "page of a book or printed manual with paragraphs",
        "paper document with margins and printed headers",
        "document photographed flat with mostly text, few graphics",
    ],
    "Whiteboard_Notes": [
        "handwritten notes on paper or whiteboard, visible marker strokes",
        "math or schematic doodles drawn by hand on paper or board",
        "notebook page with handwriting and hand-drawn diagrams",
        "whiteboard with colored markers and hand-written content",
    ],
    "Artwork_Drawing": [
        "digital illustration or hand-drawn sketch, stylized lines",
        "painting or artistic composition, not a real photograph",
        "cartoon or anime style artwork with clear stylization",
        "vector art or poster-style illustration, non-photorealistic",
    ],
    "Vehicle": [
        "photo of a car, motorcycle, train, or airplane in a real scene",
        "vehicle close-up showing wheels, lights, or brand markings",
        "transportation photo taken outdoors with vehicle as subject",
    ],
    "Indoor_Room": [
        "photo of interior room such as bedroom, living room or kitchen",
        "indoor furniture and decor visible in photo",
        "office workspace desk photo with monitor and keyboard",
    ],
    "Outdoor_Event": [
        "crowd at concert or outdoor festival with stage or tents",
        "sports event photo with many spectators or players visible",
        "public gathering or parade in an outdoor setting",
    ],
    "Electronics": [
        "printed circuit board (PCB) with visible components and solder joints, green solder mask",
        "close-up macro photo of SMD components, resistors, capacitors, ICs",
        "microcontroller development board (Arduino, STM32) on a desk with wires",
        "breadboard prototype with jumper wires, discrete components and modules",
        "electronics lab bench with multimeter, probes, power supply and PCB",
        "schematic diagram or circuit drawing with symbols and connection lines",
        "oscilloscope display or waveform screenshot from test equipment",
        "PCB layout or board 3D render showing traces and silkscreen",
        "power electronics board with large capacitors, inductors, and heat sinks",
    ],
    "Other": [
        "image that does not clearly match other categories, ambiguous or mixed content",
        "abstract, low-detail, or corrupted image where classification is uncertain",
    ],
}

# --- prepare flattened prompt list and mapping prompt -> category ---
prompts = []
prompt_to_category = {}
for category, p_list in file_promt.items():
    for p in p_list:
        prompt_to_category[p] = category
        prompts.append(p)

# create category folders if missing
for category in file_promt.keys():
    os.makedirs(os.path.join(FILE_DIRECTORY, category), exist_ok=True)

# --- collect image files (unchanged) ---
path_list = []
if RECURSIVE:
    walker = os.walk(FILE_DIRECTORY)
else:
    walker = [(FILE_DIRECTORY, [], os.listdir(FILE_DIRECTORY))]

for root, _, files in walker:
    for fname in files:
        full_path = os.path.join(root, fname)
        if os.path.isfile(full_path):
            _, ext = os.path.splitext(fname)
            if ext.lower() in image_extensions:
                path_list.append(full_path)

print(f"Found {len(path_list)} images to process.")

# --- OPENCLIP: load model + preprocess + tokenizer ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Choose model_name to fit your GPU. ViT-H-14 is best accuracy (heavy).
model_name = "ViT-H-14"    # change to "ViT-L-14" if low VRAM
pretrained = "laion2b_s32b_b79k"  # common pretrained checkpoint

model, preprocess, _ = open_clip.create_model_and_transforms( # FIXED: swapped preprocess assignment
    model_name,
    pretrained=pretrained
)
tokenizer = open_clip.get_tokenizer(model_name)

model.to(device)
model.eval()

# tokenize text once (open_clip tokenizer returns torch tensor)
text_tokens = tokenizer(prompts).to(device)  # shape: (n_prompts, seq_len)

# encode text features once and normalize
with torch.no_grad():
    text_features = model.encode_text(text_tokens)           # (n_prompts, d)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# helper to ensure unique filenames
def unique_dest_path(dest_dir, base_name):
    basename, ext = os.path.splitext(base_name)
    candidate = base_name
    i = 1
    while os.path.exists(os.path.join(dest_dir, candidate)):
        candidate = f"{basename}_{i}{ext}"
        i += 1
    return os.path.join(dest_dir, candidate)

# --- process images (uses open_clip encoding + similarity) ---
# Added tqdm for progress bar
for path in tqdm(path_list, desc="Classifying"):
    try:
        # skip if already inside category folder (optional)
        rel = os.path.relpath(path, FILE_DIRECTORY)
        parts = rel.split(os.sep)
        if len(parts) >= 2 and parts[0] in file_promt.keys():
            # Uncomment below if you want to skip verbose logging
            # print(f"Skipping (already in category): {path}")
            continue

        # preprocess & encode image
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)  # Ensure RGB
        with torch.no_grad():
            image_features = model.encode_image(img)                 # (1, d)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # compute logits like CLIP: scale by logit_scale if available
            if hasattr(model, "logit_scale"):
                logit_scale = model.logit_scale.exp()
            else:
                logit_scale = 1.0

            logits_per_image = (image_features @ text_features.t()) * logit_scale  # (1, n_prompts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        max_index = int(np.argmax(probs))
        best_prompt = prompts[max_index]
        best_category = prompt_to_category[best_prompt]
        dest_folder = os.path.join(FILE_DIRECTORY, best_category)
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = unique_dest_path(dest_folder, os.path.basename(path))

        shutil.copy2(path, dest_path)
        # Optional: Verbose logging can be slow, only print errors or summary
        # print(f"Copied: {os.path.basename(path)} -> {best_category} (score={probs[max_index]:.4f})")
        
    except Exception as e:
        print(f"\nError processing {path}: {e}")

print("Organization complete.")