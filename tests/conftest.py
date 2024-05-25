import pytest
from fastapi.testclient import TestClient
from app import app
from utils import load_model, preprocess
from PIL import Image
import numpy as np
import torch


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def sample_image():
    # Create a sample image (replace with actual image loading if needed)
    image = Image.open("tests/sample.png")
    return image


@pytest.fixture(scope="module")
def preprocessed_image(sample_image):
    return preprocess(sample_image)


@pytest.fixture(scope="module")
def high_light_image():
    image = Image.open("tests/enhanced.png")
    return image