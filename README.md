# YOLOv5 Object Detection with Custom Dataset

This repository contains code for performing object detection using YOLOv5 with a custom dataset. It includes steps for collecting and labeling images, training the YOLOv5 model, and performing real-time object detection using a webcam.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Collecting and Labeling Images](#collecting-and-labeling-images)
  - [Training the Model](#training-the-model)
  - [Real-time Object Detection](#real-time-object-detection)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Install PyTorch and other required dependencies:

```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
!git clone https://github.com/ultralytics/yolov5
!cd yolov5 && pip install -r requirements.txt
```

2. Install additional dependencies for labeling images:

```bash
!pip install pyqt5 lxml --upgrade
!cd labelImg && pyrcc5 -o libs/resources.py resources.qrc
