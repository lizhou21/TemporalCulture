import os
import argparse
import gc
import yaml
import json
import torch
import random
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    """Set random seed to ensure reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path="config.yaml"):
    """Load configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_image(image_path, target_size=(1024, 1024)):
    """Prepare the image for processing"""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize(target_size, resample=Image.LANCZOS)
    return image, original_size


def initialize_pipeline(model_path, device="cuda"):
    """Initialize Stable Diffusion XL img2img pipeline"""
    logger.info(f"Loading Stable Diffusion XL model from {model_path}")

    try:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
        pipe = pipe.to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        if device == "cuda":
            pipe.enable_attention_slicing()

        return pipe
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def process_image(item, pipe, config, output_dir, device="cuda"):
    """Process a single image item"""
    image_path = item["image_path"]
    modified_caption = item["modified_caption"]

    try:
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"modern_{image_name}")
        original_output_path = os.path.join(output_dir, f"original_{image_name}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        target_size = (1024, 1024)  # SDXL optimized size
        init_image, original_size = prepare_image(image_path, target_size)

        init_image.save(original_output_path)

        if len(modified_caption) > 150:
            logger.warning(f"Caption too long, truncating to 150 characters")
            modified_caption = modified_caption[:150]

        strength = config.get("strength", 0.6)
        guidance_scale = config.get("guidance_scale", 7.5)
        num_inference_steps = config.get("num_inference_steps", 50)
        negative_prompt = config.get("negative_prompt", "low quality, blurry")

        aesthetic_score = config.get("aesthetic_score", 6.0)
        negative_aesthetic_score = config.get("negative_aesthetic_score", 2.5)

        logger.info(f"Processing image {image_path} with caption: {modified_caption[:50]}...")
        logger.info(f"Original size: {original_size}, Target size: {target_size}")

        with torch.autocast(device_type='cuda', dtype=torch.float32) if device == "cuda" else torch.no_grad():
            result = pipe(
                prompt=modified_caption,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                original_size=original_size,
                target_size=target_size,
                aesthetic_score=aesthetic_score,
                negative_aesthetic_score=negative_aesthetic_score,
            )
            result.images[0].save(output_path)

        logger.info(f"Generated image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Use Stable Diffusion XL for image generation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--captions", type=str, help="JSON file containing descriptions (overrides config settings)")
    parser.add_argument("--output", type=str, help="Output directory (overrides config settings)")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--index", type=int, default=None, help="Process specific image index (starting from 0)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config settings)")
    parser.add_argument("--strength", type=float, default=None, help="Image retention strength (0-1)")
    parser.add_argument("--aesthetic_score", type=float, default=None, help="Aesthetic score (1-10)")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.strength is not None:
        config["strength"] = args.strength
    if args.aesthetic_score is not None:
        config["aesthetic_score"] = args.aesthetic_score

    seed_value = args.seed if args.seed is not None else config.get("seed", 42)
    seed_everything(seed_value)
    logger.info(f"Using random seed: {seed_value}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    captions_file = args.captions if args.captions else config.get("captions_file", "./captions.json")
    if not os.path.exists(captions_file):
        logger.error(f"Descriptions file does not exist: {captions_file}")
        return

    output_dir = args.output if args.output else config.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(captions_file, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load descriptions file: {e}")
        return

    logger.info(f"Loaded {len(captions_data)} image descriptions from {captions_file}")

    if args.index is not None:
        if 0 <= args.index < len(captions_data):
            captions_data = [captions_data[args.index]]
            logger.info(f"Only processing image at index {args.index}: {captions_data[0]['image_path']}")
        else:
            logger.error(f"Index {args.index} out of range 0-{len(captions_data) - 1}")
            return

    model_path = "path/to/your/model"
    try:
        pipe = initialize_pipeline(model_path, device)
    except Exception as e:
        logger.error(f"Failed to initialize model, cannot proceed: {e}")
        return

    success_count = 0

    try:
        for item in tqdm(captions_data, desc="Processing images"):
            if process_image(item, pipe, config, output_dir, device):
                success_count += 1
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"Processing complete! Success: {success_count}/{len(captions_data)}")
    print(f"\nProcessing complete! Success: {success_count}/{len(captions_data)}")
    print(f"Generated images saved to: {output_dir}")


if __name__ == "__main__":
    main()
