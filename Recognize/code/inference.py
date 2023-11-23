import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram
from ram import inference_ram
from ram import get_transform

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ram(pretrained=model_dir, image_size=384, vit='swin_l')
    model.eval()
    model = model.to(device)
    return model

def input_fn(request_body, request_content_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size=384)
    image = transform(Image.open(request_body)).unsqueeze(0).to(device)
    return image

def predict_fn(input_data, model):
    res = inference_ram(input_data, model)
    return res[0]

def output_fn(prediction, content_type):
    print("Image Tags: ", prediction)
    return prediction

def main():
    model_fn("pretrained/ram_plus_swin_large_14m.pth")

if __name__ == "__main__":
    main()