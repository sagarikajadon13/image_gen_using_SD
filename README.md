# Stable Diffusion with ControlNet

This repository contains a script to generate images using Stable Diffusion and ControlNet models. The script processes a depth maps to generate different variations of images.

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
git clone https://github.com/yourusername/stable-diffusion-controlnet.git
cd stable-diffusion-controlnet
```


To run the script with a specific depth image path and prompt, use:

```sh
python generate_images.py --image_path /path/to/depth_image --prompt your_prompt
```
