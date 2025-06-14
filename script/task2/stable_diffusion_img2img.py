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
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path="config.yaml"):
    """Load configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_image(image_path, target_size=(768, 768)):
    """Prepare image for processing"""
    image = Image.open(image_path).convert("RGB")

    # Resize while maintaining aspect ratio
    width, height = image.size
    if width > height:
        new_width = target_size[0]
        new_height = int(height * new_width / width)
    else:
        new_height = target_size[1]
        new_width = int(width * new_height / height)

    image = image.resize((new_width, new_height), resample=Image.LANCZOS)

    # Center crop or pad to exact target size
    if new_width != target_size[0] or new_height != target_size[1]:
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        new_img.paste(image, (paste_x, paste_y))
        image = new_img

    return image


def create_long_clip_text_encoder(model_path, max_length=77):
    """Create CLIP text encoder with support for longer text inputs"""
    logger.info(f"Creating Long-CLIP text encoder, max length: {max_length}")

    try:
        # Load tokenizer and text encoder config
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder_config = CLIPTextConfig.from_pretrained(model_path, subfolder="text_encoder")

        # Modify max position embeddings length
        text_encoder_config.max_position_embeddings = max_length

        # Initialize text encoder with new config
        long_text_encoder = CLIPTextModel(text_encoder_config)

        # Load original weights
        original_text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

        # Copy token embeddings
        long_text_encoder.text_model.embeddings.token_embedding.weight.data[
        :original_text_encoder.text_model.embeddings.token_embedding.weight.data.shape[
            0]] = original_text_encoder.text_model.embeddings.token_embedding.weight.data

        # Copy and extend position embeddings
        orig_position_embeddings = original_text_encoder.text_model.embeddings.position_embedding.weight.data
        position_embeddings = torch.nn.Embedding(max_length, text_encoder_config.hidden_size)
        position_embeddings.weight.data[:orig_position_embeddings.shape[0]] = orig_position_embeddings

        # Extend position embeddings if needed
        if max_length > orig_position_embeddings.shape[0]:
            position_ids = torch.arange(max_length)
            old_position_ids = torch.arange(orig_position_embeddings.shape[0])
            old_position_embeddings = orig_position_embeddings.detach().cpu().numpy()

            # Linear interpolation for each dimension
            for i in range(text_encoder_config.hidden_size):
                position_embeddings.weight.data[:, i] = torch.tensor(
                    np.interp(position_ids.cpu(), old_position_ids.cpu(), old_position_embeddings[:, i]),
                    dtype=position_embeddings.weight.data.dtype,
                )

        long_text_encoder.text_model.embeddings.position_embedding = position_embeddings

        # Copy other parameters
        long_text_encoder.text_model.encoder.load_state_dict(original_text_encoder.text_model.encoder.state_dict())
        long_text_encoder.text_model.final_layer_norm.load_state_dict(
            original_text_encoder.text_model.final_layer_norm.state_dict())

        # Update tokenizer max length
        tokenizer.model_max_length = max_length

        return tokenizer, long_text_encoder

    except Exception as e:
        logger.error(f"Failed to create Long-CLIP encoder: {e}")
        raise


def initialize_pipeline(model_path, max_length=77, device="cuda"):
    """Initialize SD img2img pipeline with Long-CLIP support"""
    logger.info(f"Loading model from {model_path}")

    try:
        # Create Long-CLIP text encoder
        tokenizer, long_text_encoder = create_long_clip_text_encoder(model_path, max_length)

        # Load other components
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

        # Create pipeline
        pipe = StableDiffusionImg2ImgPipeline(
            vae=vae,
            text_encoder=long_text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler"),
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Move to device and optimize
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.to(torch_dtype=torch.float16)

        # Enable memory optimizations
        logger.info("Enabling attention slicing optimization")
        pipe.enable_attention_slicing()

        return pipe

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def process_image(item, pipe, config, output_dir, device="cuda"):
    """Process a single image item"""
    image_path = item["image_path"]
    modified_caption = item["modified_caption"]

    logger.info(f"Processing image: {image_path}")

    try:
        # Create output path
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"sd_{image_name}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare image
        init_image = prepare_image(image_path, target_size=(768, 768))

        # Get generation parameters
        strength = config.get("strength", 0.65)
        guidance_scale = config.get("guidance_scale", 7.5)
        num_inference_steps = config.get("num_inference_steps", 50)
        negative_prompt = config.get("negative_prompt", "blurry, bad anatomy, bad hands, cropped")

        logger.info(f"Processing image {image_path}, caption length: {len(modified_caption.split())} words")

        # Generate image
        with torch.autocast(device_type='cuda', dtype=torch.float16) if device == "cuda" else torch.no_grad():
            result = pipe(
                prompt=modified_caption,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )

            # Save generated image
            result.images[0].save(output_path)

        logger.info(f"Generated image saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch image processing with Stable Diffusion (Long-CLIP version)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--captions", type=str, help="JSON file containing captions")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--index", type=int, default=None, help="Process only specific index (starting from 0)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--max_length", type=int, default=77, help="Long-CLIP maximum token length")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    seed_value = args.seed if args.seed is not None else config.get("seed", 42)
    seed_everything(seed_value)
    logger.info(f"Using random seed: {seed_value}")

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Determine input JSON file
    captions_file = args.captions if args.captions else config.get("captions_file", "./captions.json")
    if not os.path.exists(captions_file):
        logger.error(f"Captions file does not exist: {captions_file}")
        return

    # Determine output directory
    output_dir = args.output if args.output else config.get("output_dir", "./output_longclip")
    os.makedirs(output_dir, exist_ok=True)

    # Load captions
    try:
        with open(captions_file, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except Exception as e:
        logger.error(f"Unable to load captions file: {e}")
        return

    logger.info(f"Loaded {len(captions_data)} image captions from {captions_file}")
    logger.info(f"Using Long-CLIP with max token length: {args.max_length}")

    # Process specific index if specified
    if args.index is not None:
        if 0 <= args.index < len(captions_data):
            captions_data = [captions_data[args.index]]
            logger.info(f"Processing only index {args.index}: {captions_data[0]['image_path']}")
        else:
            logger.error(f"Index {args.index} out of range 0-{len(captions_data) - 1}")
            return

    # Initialize model
    model_path = config.get("sd_path", "runwayml/stable-diffusion-v1-5")

    try:
        pipe = initialize_pipeline(model_path, args.max_length, device)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    # Process all images
    success_count = 0

    try:
        for item in tqdm(captions_data, desc="Processing images"):
            if process_image(item, pipe, config, output_dir, device):
                success_count += 1

            # Clean GPU memory
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        # Cleanup
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"Processing complete! Success: {success_count}/{len(captions_data)}")
    print(f"\nProcessing complete! Success: {success_count}/{len(captions_data)}")
    print(f"Generated images saved to: {output_dir}")


if __name__ == "__main__":
    main()
