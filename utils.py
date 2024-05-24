import yaml
import numpy as np
import torch
import model

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

scale_factor = config['SCALE_FACTOR']


def preprocess(image):
    # Apply transformations
    image = (np.asarray(image) / 255.0)
    image = torch.from_numpy(image).float()
    h = (image.shape[0] // scale_factor) * scale_factor
    w = (image.shape[1] // scale_factor) * scale_factor
    image = image[0:h, 0:w, :]
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)

    return image

def load_model():
    DCE_net = model.enhance_net_nopool(scale_factor)
    DCE_net.load_state_dict(torch.load('Epoch99.pth', map_location=torch.device('cpu')))
    DCE_net.eval()

    return DCE_net