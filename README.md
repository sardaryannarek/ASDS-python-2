
---

# Low-Light Image Enhancement with Zero-DCE

This project implements a web application for enhancing low-light images using the Zero-Reference Deep Curve Estimation (Zero-DCE) model. The application allows users to upload an image and receive an enhanced version of it. 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

## Introduction

The Zero-DCE model is designed to enhance low-light images without any reference image. It employs deep learning techniques to automatically adjust the illumination of an image while preserving its natural look. This implementation uses FastAPI for the web interface and PyTorch for the deep learning model.

## Features

- Upload an image to be enhanced
- View the original and enhanced images
- View the pipeline and network details
- Simple and intuitive web interface

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/sardaryannarek/low-light-enhancement.git
    cd low-light-enhancement
    ```

2. **Create a virtual environment and activate it:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application:**

    ```sh
    uvicorn app:app --reload
    ```

2. **Open your browser and navigate to:**

    ```
    http://localhost:8000
    ```

3. **Upload an image and get the enhanced version:**
    - Click the "Upload and Enhance" button after selecting an image.

## Project Structure

```
├── static
│   ├── index.html
│   ├── network.png
│   ├── pedro.gif
│   ├── pipeline.html
│   ├── pipeline.png
│   └── thank_you.html
├── app.py
├── config.yml
├── Epoch99.pth
├── model.py
├── output.png
├── README.md
└── utils.py
```

- **app.py:** Main application file containing the FastAPI routes and prediction logic.
- **config.yml:** Configuration file (if needed).
- **model.py:** Defines the model architecture and loading mechanism.
- **utils.py:** Utility functions for preprocessing and model loading.
- **static/:** Contains static files such as HTML templates, images, and CSS.

## References

- Guo, C., Li, C., Guo, J., Loy, C. C., Hou, J., Kwong, S., & Cong, R. (2020). [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf). Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, 1780-1789.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---