import os
import argparse
import numpy as np
import json
import torch
import yaml
import logging
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Load configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_long_clip_text_encoder(pretrained_model_name_or_path, max_length=256):
    logger.info(f"Using Long-CLIP, max length: {max_length}")

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

    text_encoder_config = CLIPTextConfig.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_config.max_position_embeddings = max_length

    long_text_encoder = CLIPTextModel(text_encoder_config)

    original_text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")

    long_text_encoder.text_model.embeddings.token_embedding.weight.data[
    :original_text_encoder.text_model.embeddings.token_embedding.weight.data.shape[
        0]] = original_text_encoder.text_model.embeddings.token_embedding.weight.data

    orig_position_embeddings = original_text_encoder.text_model.embeddings.position_embedding.weight.data
    position_embeddings = torch.nn.Embedding(max_length, text_encoder_config.hidden_size)
    position_embeddings.weight.data[:orig_position_embeddings.shape[0]] = orig_position_embeddings
    if max_length > orig_position_embeddings.shape[0]:
        position_ids = torch.arange(max_length)
        old_position_ids = torch.arange(orig_position_embeddings.shape[0])
        old_position_embeddings = orig_position_embeddings.detach().cpu().numpy()
        for i in range(text_encoder_config.hidden_size):
            position_embeddings.weight.data[:, i] = torch.tensor(
                np.interp(position_ids.cpu(), old_position_ids.cpu(), old_position_embeddings[:, i]),
                dtype=position_embeddings.weight.data.dtype,
            )

    long_text_encoder.text_model.embeddings.position_embedding = position_embeddings

    long_text_encoder.text_model.encoder.load_state_dict(original_text_encoder.text_model.encoder.state_dict())
    long_text_encoder.text_model.final_layer_norm.load_state_dict(
        original_text_encoder.text_model.final_layer_norm.state_dict())

    tokenizer.model_max_length = max_length

    return tokenizer, long_text_encoder


def initialize_pipeline(max_length=256, device="cuda"):
    logger.info(f"Initialize the InstructPix2Pix model...")
    logger.info(f"Using Long-CLIP, Max token length: {max_length}")

    try:
        tokenizer, long_text_encoder = create_long_clip_text_encoder("your_model_path", max_length)

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "your_model_path",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            text_encoder=long_text_encoder,
            tokenizer=tokenizer,
        ).to(device)

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        if device == "cuda":
            pipe.enable_attention_slicing()

        return pipe
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def process_image(item, pipe, config, output_dir, device="cuda"):
    image_path = item["image_path"]

    modified_caption = item.get("modified_caption", "")

    if not modified_caption:
        logger.warning(f"Image {image_path} has no modified_caption, skipping processing")
        return False

    logger.info(f"Processing image with full instruction: {image_path}")
    logger.info(f"Instruction length: {len(modified_caption.split())} words")

    try:
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"modern_{image_name}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        src_image = Image.open(image_path).convert("RGB")
        width, height = src_image.size
        max_size = 512
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        src_image = src_image.resize((new_width, new_height), Image.LANCZOS)

        logger.info(f"Processing image {image_path} with instruction: {modified_caption[:50]}...")

        image_guidance_scale = config.get("image_guidance_scale", 1.5)
        guidance_scale = config.get("guidance_scale", 7.5)
        num_inference_steps = config.get("num_inference_steps", 30)

        with torch.autocast(device_type='cuda', dtype=torch.float16) if device == "cuda" else torch.no_grad():
            image = pipe(
                modified_caption,
                image=src_image,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale
            ).images[0]

            image.save(output_path)

        logger.info(f"Generated image saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Use local InstructPix2Pix for image editing (Long-CLIP version)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--captions", type=str, help="JSON file containing descriptions (overrides config settings)")
    parser.add_argument("--output", type=str, help="Output directory (overrides config settings)")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--index", type=int, default=None, help="Process specific image index (starting from 0)")
    parser.add_argument("--image_guidance_scale", type=float, help="Image guidance scale (0.0-3.0)")
    parser.add_argument("--guidance_scale", type=float, help="Text guidance scale (1.0-15.0)")
    parser.add_argument("--max_length", type=int, default=256, help="Long-CLIP max token length")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.image_guidance_scale is not None:
        config["image_guidance_scale"] = args.image_guidance_scale
    if args.guidance_scale is not None:
        config["guidance_scale"] = args.guidance_scale

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    captions_file = args.captions if args.captions else config.get("captions_file", "./captions.json")
    if not os.path.exists(captions_file):
        logger.error(f"Descriptions file does not exist: {captions_file}")
        return

    output_dir = args.output if args.output else config.get("output_dir", "./output_instruct_longclip")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(captions_file, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load descriptions file: {e}")
        return

    logger.info(f"Loaded {len(captions_data)} image descriptions from {captions_file}")
    logger.info(f"Using image guidance scale: {config.get('image_guidance_scale', 0.2)}")
    logger.info(f"Using text guidance scale: {config.get('guidance_scale', 13)}")

    if args.index is not None:
        if 0 <= args.index < len(captions_data):
            captions_data = [captions_data[args.index]]
            logger.info(f"Only processing image at index {args.index}: {captions_data[0]['image_path']}")
        else:
            logger.error(f"Index {args.index} out of range 0-{len(captions_data) - 1}")
            return

    try:
        pipe = initialize_pipeline(args.max_length, device)
    except Exception as e:
        logger.error(f"Model initialization failed, cannot proceed: {e}")
        return

    success_count = 0
    for item in tqdm(captions_data, desc="Processing images"):
        if process_image(item, pipe, config, output_dir, device):
            success_count += 1

        if device == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"Processing complete! Success: {success_count}/{len(captions_data)}")
    print(f"\nProcessing complete! Success: {success_count}/{len(captions_data)}")
    print(f"Generated images saved to: {output_dir}")


if __name__ == "__main__":
    main()
