import yaml
import numpy as np
import torch
import model

# Load configuration
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

scale_factor = config['SCALE_FACTOR']


def preprocess(image):
    """
    Preprocess the image for the input to the DCE-net model.

    This function normalizes the image, converts it to a tensor, adjusts its
    dimensions to be compatible with the model, and formats it for model input.

    :param image: Input image to be preprocessed, expected to be in numpy array format.
    :return: Preprocessed image as a torch tensor with shape (1, C, H, W) ready for model input.
    """
    # Normalize the image to [0, 1]
    image = (np.asarray(image) / 255.0)

    # Convert the image to a torch tensor
    image = torch.from_numpy(image).float()

    # Adjust image dimensions to be multiples of the scale factor
    h = (image.shape[0] // scale_factor) * scale_factor
    w = (image.shape[1] // scale_factor) * scale_factor
    image = image[0:h, 0:w, :]

    # Permute the dimensions to (C, H, W) and add batch dimension
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)

    return image


def load_model():
    """
    Load the DCE-net model.

    This function loads the DCE-net model architecture, loads its pre-trained
    weights, and sets it to evaluation mode.

    :return: The loaded DCE-net model in evaluation mode.
    """
    # Instantiate the model with the specified scale factor
    DCE_net = model.enhance_net_nopool(scale_factor)

    # Load the model weights
    DCE_net.load_state_dict(torch.load('Epoch99.pth', map_location=torch.device('cpu')))

    # Set the model to evaluation mode
    DCE_net.eval()

    return DCE_net
