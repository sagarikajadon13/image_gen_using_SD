import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import time
import os
import argparse


controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, use_safetensors=True)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()

generator = torch.manual_seed(12345)

def extract_canny_edges(depth_image):

    transform = transforms.Compose([
        transforms.Resize((depth_image.height, depth_image.width)),
        transforms.ToTensor()
    ])

    depth_tensor = transform(depth_image).numpy()[0]
    depth_tensor = (depth_tensor * 255).astype(np.uint8)
    edges = cv2.Canny(depth_tensor, 100, 200)
    edges_image = Image.fromarray(edges)
    return edges_image


def depth_to_normal_map(depth_image):
    transform = transforms.Compose([
        transforms.Resize((depth_image.height, depth_image.width)),
        transforms.ToTensor()
    ])

    depth_image = transform(depth_image).numpy()[0]
    depth_image = (depth_image * 255).astype(np.uint8)
    
    
    grad_x = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=5)
    

    normals = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.float32)
    normals[..., 0] = -grad_x  
    normals[..., 1] = -grad_y  
    normals[..., 2] = 1.0  


    norm = np.linalg.norm(normals, axis=2)
    normals[..., 0] /= norm
    normals[..., 1] /= norm
    normals[..., 2] /= norm
    
    normal_map = (normals + 1.0) / 2.0  
  
    normal_map = (normal_map * 255).astype(np.uint8)
    normal_map = Image.fromarray(normal_map)
    
    return normal_map



def generate_image_with_depth_maps(prompt, depth_image):
    result = pipeline(prompt, image=depth_image, generator=generator, num_inference_steps=50, guidance=7.5, strength=0.8).images[0]
    return result

def generate_image_with_canny_edges(prompt, depth_image, canny_edges):

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()

    result = pipeline(prompt, image=canny_edges, generator=generator, num_inference_steps=50, guidance=7.5, strength=0.8).images[0]
    return result


def generate_image_with_normals(prompt, depth_image, normal_map):

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype=torch.float16, use_safetensors=True)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()

    result = pipeline(prompt, image=normal_map, generator=generator, num_inference_steps=50, guidance=7.5, strength=0.8).images[0]
    return result


def generate_images_with_aspect_ratios(prompt, depth_image):
    aspect_ratios = [(512, 512), (512, 768), (768, 512)]
    generated_images = []

    for ratio in aspect_ratios:
        width, height = ratio
        depth_image = depth_image.resize((width, height))
        result = pipeline(prompt, image=depth_image, width=width, height=height, generator=generator, num_inference_steps=50, guidance=7.5, strength=0.8).images[0]
        generated_images.append(result)

    return generated_images

def generate_image_combined(prompt, depth_image, canny_edges, normal_map):

    controlnets = [
        ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, use_safetensors=True),
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True),
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype=torch.float16, use_safetensors=True)
    ]

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnets, torch_dtype=torch.float16
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()

    result = pipeline(prompt, image=[depth_image, canny_edges, normal_map], generator=generator, num_inference_steps=50, guidance=7.5, strength=0.8).images[0]
    return result



def measure_generation_latency(prompt, depth_image, num_inference_steps):
    start_time = time.time()
    result = pipeline(prompt, image=depth_image, generator=generator, num_inference_steps=num_inference_steps, guidance=7.5, strength=0.8).images[0]
    latency = time.time() - start_time

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    start_time_eu = time.time()
    result_eu = pipeline(prompt, image=depth_image, generator=generator, num_inference_steps=num_inference_steps, guidance=7.5, strength=0.8).images[0]
    eu_latency = time.time() - start_time_eu

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    start_time_ddim = time.time()
    result_ddim = pipeline(prompt, image=depth_image, generator=generator, num_inference_steps=num_inference_steps, guidance=7.5, strength=0.8).images[0]
    ddim_latency = time.time() - start_time_ddim

    return latency, eu_latency, ddim_latency, result, result_eu, result_ddim

def main():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion with ControlNet")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the depth image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")

    args = parser.parse_args()

    depth_image_path = args.image_path
    prompt = args.prompt
  
    fname = depth_image_path.split('/')[-1]
    
    if fname.endswith(".npy"):
        depth_array = np.load(depth_image_path)
        depth_image = Image.fromarray(depth_array)
    else:
        depth_image = Image.open(depth_image_path)

    depth_image = depth_image.resize((512, 512))

    canny_edges = extract_canny_edges(depth_image)
    normal_map = depth_to_normal_map(depth_image)

    os.makedirs(f"./generated_images/{fname}", exist_ok=True)

    generated_image = generate_image_with_depth_maps(prompt, depth_image)
    generated_image.save(f"./generated_images/{fname}/depth.png")

    generated_image = generate_image_with_canny_edges(prompt, depth_image, canny_edges)
    generated_image.save(f"./generated_images/{fname}/canny_edges.png")

    generated_image = generate_image_with_normals(prompt, depth_image, normal_map)
    generated_image.save(f"./generated_images/{fname}/normal_map.png")

    generated_images = generate_images_with_aspect_ratios(prompt, depth_image)
    for i, img in enumerate(generated_images):
        img.save(f"./generated_images/{fname}/generated_image_{i}.png")

    generated_image = generate_image_combined(prompt, depth_image, canny_edges, normal_map)
    generated_image.save(f"./generated_images/{fname}/combined.png")

    num_inference_steps = 50
    latency, eu_latency, ddim_latency, result, result_eu, result_ddim = measure_generation_latency(prompt, depth_image, num_inference_steps)
    print(f"num_inference_steps = 50, Initial Latency: {latency}s, Euler's Latency: {eu_latency}s, DDIM Latency: {ddim_latency}s")

  
    result.save(f"./generated_images/{fname}/result_dpm_{num_inference_steps}.png")
    result_eu.save(f"./generated_images/{fname}/result_eu_{num_inference_steps}.png")
    result_ddim.save(f"./generated_images/{fname}/result_ddim_{num_inference_steps}.png")


    num_inference_steps = 25
    latency, eu_latency, ddim_latency, result, result_eu, result_ddim = measure_generation_latency(prompt, depth_image, num_inference_steps)
    print(f"num_inference_steps = 25, Initial Latency: {latency}s, Euler's Latency: {eu_latency}s, DDIM Latency: {ddim_latency}s")

    result.save(f"./generated_images/{fname}/result_dpm_{num_inference_steps}.png")
    result_eu.save(f"./generated_images/{fname}/result_eu_{num_inference_steps}.png")
    result_ddim.save(f"./generated_images/{fname}/result_ddim_{num_inference_steps}.png")


if __name__ == "__main__":
    main()
