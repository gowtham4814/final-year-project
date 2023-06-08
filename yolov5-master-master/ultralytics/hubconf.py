"""YoloV5 models / weights / configs"""

dependencies = ["torch", "yaml"]

import torch

from models.common import Conv, DWConv
from models.yolo import Detect, Encoder, Model, create_modules
from utils.general import (
    check_file, dataset_labels, download, is_ascii, make_divisible, non_max_suppression, plot_images, scale_coords,
    xyxy2xywh
)
from utils.torch_utils import time_sync

_MODEL_CONFIGS = {
    # Model : [config, url]
    'yolov5s': ['models/yolov5s.yaml', 'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.yaml'],
    'yolov5m': ['models/yolov5m.yaml', 'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.yaml'],
    'yolov5l': ['models/yolov5l.yaml', 'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.yaml'],
    'yolov5x': ['models/yolov5x.yaml', 'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.yaml']
}

def _create(name, pretrained, channels, classes):
    config, url = _MODEL_CONFIGS[name]
    if pretrained and 'http' in url:
        from copy import deepcopy
        from pathlib import Path
        from urllib.request import urlopen
        from models.common import fuse_conv_and_bn
        from models.yolo import attempt_load

        try:  # download/load FP32 model
            torch.hub.download_url_to_file(url, Path('./models') / config.replace('.yaml', '.pt'))
            attempt_load(f'./models/{name}.pt', map_location=torch.device('cpu'))  # to cache z=0 weights
            ckpt = torch.load(Path('./models') / config.replace('.yaml', '.pt'), map_location=torch.device('cpu'))
            for k, v in list(ckpt['model'].items()):
                if isinstance(v, torch.HalfTensor):
                    ckpt['model'][k] = v.float()
            model = Model(config, channels, classes, pretrained=False).to(torch.device('cpu'))
            model = fuse_conv_and_bn(model)
            model.load_state_dict(ckpt['model'])
            print(f"Loaded pretrained weights from '{url}'")
        except Exception as e:
            print(f"WARNING: Could not download '{url}' ({e})")
            if pretrained:  # fallback to pretrained on COCO
                print(f"WARNING: Loading pretrained weights from '{url.replace('.pt', '.pth')}'")
                ckpt = torch.load(attempt_download(url.replace('.pt', '.pth'), './checkpoints'), map_location=torch.device('cpu'))  # download
                model = Model(config, channels, classes, pretrained=False).to(torch.device('cpu'))  # create
                state_dict = ckpt['model'].float().state_dict()  # to FP32

