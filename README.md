# AI-Powered Time-lapse Generation

This project uses deep learning to create smooth time-lapse videos from a sequence of images by interpolating intermediate frames. It leverages the Film-Style frame interpolation model to generate natural transitions between input frames.
Refer:
https://github.com/google-research/frame-interpolation/tree/main

## Features

- Frame interpolation between pairs of images using a pretrained neural network
- Support for multiple input image formats (PNG, JPEG, BMP)
- Automatic image resizing while maintaining aspect ratio
- Face timelapse with automatic face centering and background removal
- Configurable output FPS and interpolation settings
- GPU acceleration support

## Setup

1. Create a conda environment with TensorFlow and GPU support. If you are on Windows, use WSL.

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pretrained model:
    ```python
    import gdown
    import os
    os.makedirs('pretrained_models/film_net/Style/saved_model', exist_ok=True)
    folder_url = 'https://drive.google.com/drive/folders/1i9Go1YI2qiFWeT5QtywNFmYAA74bhXWj'
    gdown.download_folder(folder_url, output='pretrained_models/film_net/Style/saved_model')
    ```

## Usage

### Simple Frame Interpolation

Use the `simple_interpolation.ipynb` notebook to interpolate frames between two images:
```python
from eval import interpolator, util

# Initialize the interpolator
predictor = Predictor()
predictor.setup()

# Generate intermediate frames
output_frames = predictor.predict(
    frame1="input_frames/image1.jpg",
    frame2="input_frames/image2.jpg", 
    times_to_interpolate=5,
    out_dir="output_frames",
    img_size=(1080, 720)
)
```
Example
(middle image is interpolated)
![alt interpolated](sample_data/boy_interpolated.jpg)

![boy animation](sample_data/boy.gif)

### Time-lapse Generation

Use the `time_lapse.ipynb` notebook to create a time-lapse video from multiple images:

1. Place your input images in the `input_frames/<sequence_name>` directory. The image names should be of format yyyymmdd_hhmmss.jpg (android photos default names)

2. Configure settings in the notebook:
    ```python
    input_dir = 'input_frames/sequence'
    output_dir = 'output_frames/sequence'
    fps = 24
    ```

3. Run the notebook to:
    - Preprocess images to a consistent size
    - Generate interpolated frames between each pair
    - Create the final time-lapse video

The output video will be saved as `output_frames/sequence.mp4`.

Example  
![timelapse animation](sample_data/lake_window.gif)

## Face Time Lapse
Set the preprocess_face variable to true in `time_lapse.ipynb` and `simple_interpolation` video and the images will be preprocessed to center the face and remove background to enhace the time lapse experience.

1. Configure settings in the notebook:
    ```python
    input_dir = 'input_frames/sequence'
    output_dir = 'output_frames/sequence'
    preprocess_faces = True  # for faces
    equal_intervals = True  # equal no frames between images or date based
    fps = 24
    ```
Example
![timelapse animation](sample_data/shahrukh-khan_face.gif)

## Future Enhancements
- Automatically identify date format from filenames/file metadata for time lapse.
- Improve face time lapse to handle multiple faces, and have consistent lighting.