import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import multiprocessing
from src.utils import get_function_logger

def degrade_image(image, config):
    """Apply degradations to create a low-quality version of the image."""
    logger = get_function_logger()
    
    # Downsize and upsize to introduce loss
    lr_size = (32, 32)
    image = image.resize(lr_size, Image.BICUBIC).resize((128, 128), Image.BICUBIC)
    
    # Apply additional degradations
    if random.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))
        logger.debug("Applied Gaussian blur")
    
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        logger.debug("Adjusted contrast")
    
    if 'degradations' in config and 'noise' in config['degradations']:
        noise_level = random.uniform(0, config['degradations']['noise'])
        image_array = np.array(image)
        noise = np.random.normal(0, noise_level * 255, image_array.shape)
        noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy_image)
        logger.debug(f"Added noise with level {noise_level}")
    
    return image.resize((32, 32), Image.BICUBIC)

def create_lq_image(original_img, config, index):
    """Create a low-quality image."""
    lq_image = degrade_image(original_img, config)
    lq_image = lq_image.convert('RGB')  # Ensure the image is in RGB mode
    return lq_image, index

def save_lq_image(lq_img, output_dir, index):
    """Save the LQ image."""
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
    
    # Determine the number of CPU cores to use
    total_cores = os.cpu_count()
    num_cores = max(1, int(total_cores * 0.75)) if total_cores else 1
    
    logger.debug(f"Total CPU cores available: {total_cores}")
    logger.debug(f"Using {num_cores} cores for multiprocessing")
    
    logger.debug(f"Loading high-quality image from {hq_img_path}")
    with Image.open(hq_img_path) as hq_img:
        hq_img = hq_img.convert('RGB')
        logger.debug(f"High-quality image loaded with size: {hq_img.size}")
    
        os.makedirs(output_dir, exist_ok=True)
        
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
            results = pool.starmap(save_lq_image, 
                                   [(lq, output_dir, i) for lq, i in all_lq_images])
        
    # Log results
    for result in results:
        logger.debug(result)
    
    logger.debug(f"Dataset created with {num_samples} LQ samples")

    # Create a file to distinguish between train and validation samples
    with open(os.path.join(output_dir, 'val_samples.txt'), 'w') as f:
        for i in range(num_train_samples, num_samples):
            f.write(f"lr_qr_{i}.png\n")

    logger.debug(f"Created val_samples.txt to distinguish {num_val_samples} validation samples")