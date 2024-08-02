import os
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
import random
import multiprocessing
import json
from io import BytesIO
from src.utils import get_function_logger

def degrade_image(image, config):
    """Apply focused degradations to create a low-quality version of the grayscale QR code image."""
    logger = get_function_logger()
    
    # Ensure the image is 128x128 pixels and in grayscale
    image = image.resize((128, 128), Image.BICUBIC).convert('L')
    
    # Apply blur
    blur_radius = random.uniform(0.5, config['degradations'].get('blur', 3.0))
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    logger.debug(f"Applied Gaussian blur with radius {blur_radius}")
    
    # Add noise
    noise_level = random.uniform(0.01, config['degradations'].get('noise', 0.15))
    image_array = np.array(image)
    noise = np.random.normal(0, noise_level * 255, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(noisy_image, mode='L')
    logger.debug(f"Added noise with level {noise_level}")
    
    # Apply JPEG compression
    jpeg_quality = random.randint(20, config['degradations'].get('jpeg_quality', 80))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    image = Image.open(buffer).convert('L')
    logger.debug(f"Applied JPEG compression with quality {jpeg_quality}")
    
    # Randomly apply contrast adjustment
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.7, 1.3)
        image = enhancer.enhance(factor)
        logger.debug(f"Adjusted contrast with factor {factor}")
    
    # Randomly apply brightness adjustment
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.7, 1.3)
        image = enhancer.enhance(factor)
        logger.debug(f"Adjusted brightness with factor {factor}")
    
    return image

"""
def degrade_image(image, config):
    #Apply degradations to create a low-quality version of the image.
    logger = get_function_logger()
    
    # Ensure the image is 128x128 pixels
    image = image.resize((128, 128), Image.BICUBIC)
    
    # Always apply some level of blur, noise, and JPEG compression
    # Blur: Simulates camera focus issues or motion blur
    blur_radius = random.uniform(0.5, config['degradations'].get('blur', 3.0))
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    logger.debug(f"Applied Gaussian blur with radius {blur_radius}")
    
    # Noise: Simulates sensor noise, especially in low-light conditions
    noise_level = random.uniform(0.01, config['degradations'].get('noise', 0.15))
    image_array = np.array(image)
    noise = np.random.normal(0, noise_level * 255, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(noisy_image)
    logger.debug(f"Added noise with level {noise_level}")
    
    # JPEG compression: Simulates artifacts from digital compression
    jpeg_quality = random.randint(20, config['degradations'].get('jpeg_quality', 80))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    image = Image.open(buffer)
    logger.debug(f"Applied JPEG compression with quality {jpeg_quality}")
    
    # Randomly apply additional degradations
    # Contrast adjustment: Simulates poor lighting conditions
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.7, 1.3)
        image = enhancer.enhance(factor)
        logger.debug(f"Adjusted contrast with factor {factor}")
    
    # Brightness adjustment: Simulates over or under-exposure
    if random.random() < 0.3:
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.7, 1.3)
        image = enhancer.enhance(factor)
        logger.debug(f"Adjusted brightness with factor {factor}")
    
    # Perspective distortion: Simulates QR code captured at an angle
    if random.random() < 0.2:
        width, height = image.size
        m = random.uniform(-0.1, 0.1)
        xshift = abs(m) * width
        new_width = width + int(round(xshift))
        image = image.transform((new_width, height), Image.AFFINE, 
                                (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                                Image.BICUBIC)
        image = image.resize((width, height), Image.BICUBIC)
        logger.debug(f"Applied slight perspective distortion")
    
    # Uneven lighting: Simulates shadows or uneven illumination
    if random.random() < 0.3:
        gradient = Image.new('L', image.size, color=0)
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                gradient.putpixel((x, y), int(255 * (x / image.size[0])))
        image = Image.composite(image, Image.new('RGB', image.size, 'white'), gradient)
        logger.debug("Applied uneven lighting effect")

    # Random lines: Simulates scratches or marks on the QR code
    if random.random() < 0.2:
        draw = ImageDraw.Draw(image)
        for _ in range(random.randint(1, 3)):
            start = (random.randint(0, image.size[0]), random.randint(0, image.size[1]))
            end = (random.randint(0, image.size[0]), random.randint(0, image.size[1]))
            draw.line([start, end], fill=random.randint(0, 255), width=random.randint(1, 3))
        logger.debug("Added random lines")
    
    # Ensure the final image is in RGB mode
    return image.convert('RGB')
"""

def create_lq_image(original_img, config, index):
    """Create a low-quality grayscale image."""
    lq_image = degrade_image(original_img, config)
    return lq_image, index

def save_lq_image(lq_img, output_dir, index):
    """Save the LQ grayscale image."""
    lq_path = os.path.join(output_dir, f"lr_qr_{index}.png")
    lq_img.save(lq_path)
    return f"Saved LQ image {index} to {output_dir}"

def create_dataset(config):
    logger = get_function_logger()
    logger.debug("Starting dataset creation.")
    
    hq_img_path = config['hq_img_path']
    output_dir = config['output_dir']
    num_samples = config['num_samples']
    val_split = config['val_split']
    perfect_qr = config['perfect_qr']
    
    # Create train and validation directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Determine the number of CPU cores to use
    total_cores = os.cpu_count()
    num_cores = max(1, int(total_cores * 0.95)) if total_cores else 1
    
    logger.debug(f"Total CPU cores available: {total_cores}")
    logger.debug(f"Using {num_cores} cores for multiprocessing")
    
    logger.debug(f"Loading high-quality image from {hq_img_path}")
    with Image.open(hq_img_path) as hq_img:
        hq_img = hq_img.convert('L')
        logger.debug(f"High-quality grayscale image loaded with size: {hq_img.size}")
    
        num_val_samples = int(num_samples * val_split)
        num_train_samples = num_samples - num_val_samples
    
        # Create and use a single pool for all operations
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Create LQ images
            logger.debug("Creating LQ images")
            all_lq_images = pool.starmap(create_lq_image, 
                                         [(hq_img, config, i) for i in range(num_samples)])
        
            # Save all LQ images
            logger.debug(f"Saving {num_samples} LQ samples")
            train_results = pool.starmap(save_lq_image, 
                                         [(lq, train_dir, i) for i, (lq, _) in enumerate(all_lq_images[:num_train_samples])])
            val_results = pool.starmap(save_lq_image, 
                                       [(lq, val_dir, i) for i, (lq, _) in enumerate(all_lq_images[num_train_samples:])])
    
    # Log results
    for result in train_results + val_results:
        logger.debug(result)
    
    logger.debug(f"Dataset created with {num_train_samples} train and {num_val_samples} validation samples")

    # Create dataset info JSON
    dataset_info = {
        "num_samples": num_samples,
        "num_train_samples": num_train_samples,
        "num_val_samples": num_val_samples,
        "hq_img_path": os.path.relpath(hq_img_path, start=output_dir),
        "degradations": config.get('degradations', {})
    }

    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)

    logger.debug(f"Created dataset_info.json in {output_dir}")

    return train_dir, val_dir, os.path.join(output_dir, 'dataset_info.json')