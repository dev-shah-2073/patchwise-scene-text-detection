# Patch-Based Scene Text Detection

Lightweight scene text detection using patch-level classification and regression.

## Overview

The image is resized and split into a fixed grid of patches. Each patch is classified as text/non-text and regresses a bounding box. Overlapping positive patches are merged to produce final text regions.

## Project Tree

patch-based-text-detection/
│
├── dataset/
│   ├── dataset.py              # End-to-end Dataset & DataLoader
│   ├── preprocessing.py        # Image resize, normalization, patchification
│   └── ground_truth.py         # GT generation and coordinate transforms
│
├── model/
│   ├── mobilenet.py            # MobileNetV3-based detection model
│   └── loss.py                 # Classification + conditional regression loss
│
├── training/
│   └── train.py                # Training loop and checkpointing
│
├── inference/
│   └── predict.py              # Patch inference, box merging, visualization
│
├── utils/
│   ├── box_ops.py              # IoU, grouping, box merging utilities
│   └── visualize.py            # Cropping and plotting helpers
│
├── configs/
│   └── config.py               # Hyperparameters and paths
│
├── notebooks/
│   └── experiments.ipynb       # Original research and debugging notebook
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore

## Training

python train.py

## Inference

python run.py

## Model

* Backbone: **MobileNetV3-Large**
* Patch size: **48×48**
* Output per patch: **[text score, x, y, w, h]**
* Loss: **BCE (classification) + conditional L1 (regression)**

## Demo

### Input image
<br/>

<img width="387" height="409" alt="Screenshot 2026-02-08 at 2 22 48 AM" src="https://github.com/user-attachments/assets/8a5b1243-c732-4e35-9c3e-6b3f6e233dee" />
<br/>

### Text Detection
<br/>
<img width="254" height="216" alt="Screenshot 2026-02-08 at 2 22 57 AM" src="https://github.com/user-attachments/assets/939f2218-0dcc-4f03-b88f-be2dc0259d8a" />
