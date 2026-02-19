## 📂 CLIP Image Organizer

This script automatically classifies images into categories using OpenAI CLIP.

It scans a folder, analyzes each image, and predicts which category it belongs to (Selfie, Nature, Food, Screenshot, or Document).

## 🚀 Features

Uses CLIP (ViT-B/32) model

Supports recursive folder scanning

Works on CPU or CUDA (GPU)

Supports common image formats (.jpg, .png, .bmp, .gif, .tiff)

## 📦 Requirements

Install dependencies:

pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install pillow numpy


CLIP Paper:
https://arxiv.org/abs/2103.00020

## ⚙️ Configuration

Edit these variables in the script:

FILE_DIRECTORY = r"your\image\folder\path"
RECURSIVE = True


You can also modify file_promt to change or improve categories.

## ▶️ Usage

Run:

python your_script_name.py


The script will print:

Image path

Probability scores

Predicted label

## 🧠 Categories

Selfie

Nature

Food

ScreenShots

Document

You can easily add more categories by editing file_promt.