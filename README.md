Image Processing with tread tools designed for image processing tasks utilizing threading to enhance performance.
This repository is designed to provide tools dedicated to image processing. An artificial intelligence component will be added soon

- ImageProcessingThread.py: Implements image processing operations within a separate thread to avoid blocking the main application. It handles loading images, applying filters or transformations, and saving the results asynchronously to improve responsiveness.

- --

- And folder /model

## Description of Super-Resolution Models

These models are pre-trained deep learning networks for **image super-resolution**, designed to upscale low-resolution images by factors of 2, 3, 4, or 8 while enhancing image quality.

| Model Name                | Description                                                                                                           |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **EDSR_x2.pb / x3 / x4**  | EDSR (Enhanced Deep Super-Resolution) is a state-of-the-art deep residual network for super-resolution. It removes unnecessary modules from traditional residual networks to improve performance and accuracy. These models upscale images by factors of 2, 3, or 4 with high fidelity and sharpness. |
| **ESPCN_x2.pb / x3 / x4** | ESPCN (Efficient Sub-Pixel Convolutional Neural Network) uses sub-pixel convolution layers to perform upscaling only at the final stage, reducing computation. It achieves real-time super-resolution with good quality, especially for smaller scaling factors (x2, x3, x4). It is lightweight and fast, suitable for video and live applications. |
| **FSRCNN-small_x2.pb / x3 / x4** | FSRCNN (Fast Super-Resolution Convolutional Neural Network) small versions are compact models optimized for speed and efficiency. They perform super-resolution with lower computational cost, suitable for devices with limited resources, upscaling by factors of 2, 3, or 4. |
| **FSRCNN_x2.pb / x3 / x4** | FSRCNN full versions are deeper networks than the small variants, offering better image quality at the cost of more computation. They upscale images by 2x, 3x, or 4x with faster inference than older models like SRCNN. |
| **LapSRN_x2.pb / x4 / x8** | LapSRN (Laplacian Pyramid Super-Resolution Network) progressively reconstructs high-resolution images through a series of convolutional layers arranged in a Laplacian pyramid structure. It supports larger upscaling factors (x2, x4, x8) and provides visually pleasing results with fewer artifacts. |


These models are typically used in image enhancement applications such as photo editing, video streaming, and computer vision preprocessing.



