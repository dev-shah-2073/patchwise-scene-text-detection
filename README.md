# Patch-Based Scene Text Detection

Lightweight scene text detection using patch-level classification and regression.

## Overview

The image is resized and split into a fixed grid of patches. Each patch is classified as text/non-text and regresses a bounding box. Overlapping positive patches are merged to produce final text regions.

## Model

* Backbone: **MobileNetV3-Large**
* Patch size: **48×48**
* Output per patch: **[text score, x, y, w, h]**
* Loss: **BCE (classification) + conditional L1 (regression)**

