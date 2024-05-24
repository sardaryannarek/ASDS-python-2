import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

def test_main_page(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "Zero-DCE: Low-Light Image Enhancement" in response.text

def test_predict_endpoint(client: TestClient, sample_image: Image):
    byte_arr = io.BytesIO()
    sample_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()

    response = client.post(
        "/predict/",
        files={"file": ("filename", byte_arr, "image/png")}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert "original_filename" in json_response
    assert "enhanced_filename" in json_response

def test_pipeline_page(client: TestClient):
    response = client.get("/pipeline")
    assert response.status_code == 200
    assert "Light-enhancement curves" in response.text

def test_thank_you_page(client: TestClient):
    response = client.get("/thank-you")
    assert response.status_code == 200
    assert "Thank You" in response.text