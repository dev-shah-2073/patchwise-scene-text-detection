# Patch-Based Scene Text Detection

Lightweight scene text detection using patch-level classification and regression.

## Overview

The image is resized and split into a fixed grid of patches. Each patch is classified as text/non-text and regresses a bounding box. Overlapping positive patches are merged to produce final text regions.

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
