"""
Step 1: Generate and modify descriptions for Hanfu images and save the results to a JSON file.
Uses GPT-4o for image analysis and description modification.
"""

import os
import argparse
import yaml
import json
import logging
import base64
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from PIL import Image

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Hanfu description and modernization prompts
HANFU_DESCRIPTION_PROMPT = "提供这件传统中国汉服服装的详细描述，包括其设计元素、颜色、图案和特色。请详细描述衣服的颜色、形制、具体元素和样式特点。"

HANFU_MODERNIZATION_PROMPT = """
Transform this traditional Hanfu description: "{caption}" into a modern clothing design by strategically incorporating its key elements. Your task is to:

1. Choose ONE specific modern garment type as the base (hoodie, blazer, casual wear, sportswear, trench coat, streetwear, or business attire).

2. Identify 1-3 distinctive elements from the original Hanfu description (such as collar style, sleeve design, waist details, fabric patterns, or color schemes).

3. Describe exactly how these Hanfu elements are integrated into the modern garment:
   - WHERE each element is placed on the modern garment
   - HOW each element is adapted to suit contemporary fashion
   - WHY these particular elements were chosen (cultural significance)

4. Ensure the final design:
   - Is primarily a modern, wearable garment for everyday contexts
   - Emphasizes modern clothing style design
   - Displays Hanfu inspiration through intentional design choices
   - Balances contemporary style with traditional Chinese aesthetics
   - Appeals to modern fashion sensibilities while honoring cultural heritage

The output should provide a detailed description of the modern clothing design, suitable for image generation.
IMPORTANT: Final description MUST be under 77 tokens for CLIP model !!!
"""

def load_config(config_path="config.yaml"):
    """Load configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def initialize_client(config):
    """Initialize OpenAI client"""
    logger.info("Initializing OpenAI client...")
    
    try:
        api_key = config.get("openai_api_key")
        api_base = config.get("openai_api_base")
        client = OpenAI(api_key=api_key, base_url=api_base)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise

def encode_image_to_base64(image_path):
    """Encode image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption_with_gpt4o(image_path, client, config):
    """Generate detailed description for Hanfu image using GPT-4o"""
    logger.info(f"Generating description for image {image_path} using GPT-4o...")
    
    try:
        base64_image = encode_image_to_base64(image_path)
        
        response = client.chat.completions.create(
            model=config.get("model_name"),
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": HANFU_DESCRIPTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
            }],
            max_tokens=300,
            temperature=0.7
        )
        
        caption = response.choices[0].message.content.strip()
        logger.info(f"Generated description: {caption}")
        return caption
    except Exception as e:
        logger.error(f"Failed to generate description: {e}")
        return f"Failed to generate description: {str(e)}"

def modify_caption_with_llm(caption, config, client):
    """Modify caption using LLM to merge traditional Hanfu elements with modern fashion"""
    logger.info("Modifying description by merging traditional and modern elements...")
    
    prompt = HANFU_MODERNIZATION_PROMPT.format(caption=caption)
    
    try:
        response = client.chat.completions.create(
            model=config.get("model_name"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        modified_caption = response.choices[0].message.content.strip()
        logger.info(f"Modified description: {modified_caption}")
        return modified_caption
    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        return caption

def process_image(image_path, client, config):
    """Process a single image: generate and modify description"""
    try:
        caption = generate_caption_with_gpt4o(image_path, client, config)
        modified_caption = modify_caption_with_llm(caption, config, client)
        
        return {
            "image_path": str(image_path),
            "original_caption": caption,
            "modified_caption": modified_caption
        }
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return {
            "image_path": str(image_path),
            "original_caption": f"Processing failed: {str(e)}",
            "modified_caption": f"Processing failed: {str(e)}"
        }

def find_images(data_dir, extensions=None):
    """Find all image files in the specified directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(Path(data_dir).glob(f"*{ext}")))
        image_paths.extend(list(Path(data_dir).glob(f"**/*{ext}")))  # Search subdirectories
    
    return sorted(list(set(image_paths)))  # Remove duplicates and sort

def main():
    parser = argparse.ArgumentParser(description="Step 1: Generate and modify Hanfu image descriptions")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--data_dir", type=str, help="Directory of images (overrides config)")
    parser.add_argument("--output", type=str, help="Output JSON file path (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine data directory
    data_dir = args.data_dir if args.data_dir else config.get("data_dir")
    if not data_dir or not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        return
    
    # Determine output JSON file
    output_json = args.output if args.output else config.get("captions_file", "./captions.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    
    # Initialize OpenAI client
    try:
        client = initialize_client(config)
    except Exception as e:
        logger.error(f"Client initialization failed, cannot proceed: {e}")
        return
    
    # Find all image files
    logger.info(f"Finding image files in {data_dir}...")
    image_paths = find_images(data_dir)
    
    if not image_paths:
        logger.error(f"No image files found in {data_dir}")
        return
    
    logger.info(f"Found {len(image_paths)} image files")
    
    # Process all images
    results = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        result = process_image(image_path, client, config)
        results.append(result)
    
    # Save results to JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"All descriptions saved to {output_json}")
    print(f"\nProcessing complete! Processed {len(results)} images, results saved to {output_json}")

if __name__ == "__main__":
    main()
