# PEX ML example

Decide between indoor and outdoor images extracted from youtube 8m dataset.

## Dataset

Videos are downloaded with bash scripts adopted from [https://github.com/gsssrao/youtube-8m-videos-frames](https://github.com/gsssrao/youtube-8m-videos-frames).
100 videos is downloaded for each category. Frame images are extracted once per 5 second from each video.

### Indoor categories:
- Bedroom
- Bathroom
- Classroom
- Office

### Outdoor categories
- Landscape
- Skyscraper
- Mountain
- Beach

## Installation

Install youtube-dl, ffmpeg and curl into your OS. Install python requirements ``pip install -r requirements.txt``.

## Training

For training run train.sh script. It downloads videos, extract frames and creates train/test balanced file lists. Default ration is 80% for training and 20% for testing.
After that starts training itself. Trained model is stored in ckpt folder.  

## Inference

For testing image run ``evaluate.sh <model_file> <image_file>``. Result is written to stdout as Indoor or Outdoor image.
Example pretrained model file is in ckpt folder.

## Unit tests

Run ``pytest test`` in root directory.