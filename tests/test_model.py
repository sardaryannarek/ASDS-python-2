from PIL import ImageChops
import torch
from torchvision import transforms


def test_model_inference_speed(model, preprocessed_image):
    import time
    start_time = time.time()
    with torch.no_grad():
        enhanced_image, _ = model(preprocessed_image)
    end_time = time.time()
    inference_time = end_time - start_time
    assert inference_time < 5


def test_correct_output(high_light_image, preprocessed_image,model):
    with torch.no_grad():
        output, _ = model(preprocessed_image)
    output = transforms.ToPILImage()(output.squeeze())
    diff = ImageChops.difference(output, high_light_image)
    assert not diff.getbbox()

