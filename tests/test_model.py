import pytest
from PIL import Image
from PIL import ImageChops
import numpy as np
from utils import preprocess, load_model
import torch
from torchvision import transforms


def test_model_output(sample_image: Image):
    model = load_model()
    preprocessed = preprocess(sample_image)

    with torch.no_grad():
        enhanced_image, _ = model(preprocessed)
    output_image = transforms.ToPILImage()(enhanced_image.squeeze())
    real_enhanced = Image.open("tests/enhanced.png")
    diff = ImageChops.difference(output_image, real_enhanced)
    assert not diff.getbbox()


def test_model_inference_speed(model, preprocessed_image):
    import time
    start_time = time.time()
    with torch.no_grad():
        enhanced_image, _ = model(preprocessed_image)
    end_time = time.time()
    inference_time = end_time - start_time
    assert inference_time < 5
