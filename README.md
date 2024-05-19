# Image Generation Using Stable Diffusion and Controlnet

This repository contains a script to generate images using Stable Diffusion and ControlNet models. The script processes a depth map to generate different variations of images.

## Requirements

- Python 3.7 or higher
- PyTorch
- diffusers
- accelerate
- transformers
- safetensors 
- PIL (Pillow)
- numpy
- torchvision
- opencv-python
- argparse
  

## Installation

First, clone the repository:

```sh
git clone https://github.com/sagarikajadon13/image_gen_using_SD.git
cd image_gen_using_SD
```
Then install the required packages

To run the script with a specific depth image path and prompt, use:

```sh
python generation.py --image_path /path/to/depth_image --prompt your_prompt
```
