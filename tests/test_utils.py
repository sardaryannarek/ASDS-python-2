import pytest
from PIL import Image
import numpy as np
from utils import preprocess, load_model
import torch

def test_preprocess(sample_image: Image):
    processed = preprocess(sample_image)
    assert processed.shape == (1, 3, 396, 600)

def test_load_model():
    model = load_model()
    assert model is not None
    assert callable(getattr(model, 'forward', None))

