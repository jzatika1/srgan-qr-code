import qrcode
import os
import numpy as np
from PIL import Image
from src.utils import get_function_logger, ensure_directories_exist

def generate_qr_code(config):
    logger = get_function_logger()
    logger.debug("Starting QR code generation.")
    
    file_path = config['file_path']
    ensure_directories_exist([os.path.dirname(file_path)])
    
    message = config['message']
    target_size = (128, 128)
    
    # Start with the config values
    box_size = config['box_size']
    border = config['border']
    
    while True:
        qr = qrcode.QRCode(
            version=config['version'],
            error_correction=getattr(qrcode.constants, f'ERROR_CORRECT_{config["error_correction"]}'),
            box_size=box_size,
            border=border
        )
        qr.add_data(message)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color='black', back_color='white')
        
        if img.size[0] <= target_size[0] and img.size[1] <= target_size[1]:
            break
        
        # Reduce box_size and border if the image is too large
        box_size -= 1
        if box_size < 1:
            border -= 1
            box_size = config['box_size']  # Reset to original value
        
        if border < 0:
            logger.error("Cannot generate QR code within 128x128 size limit.")
            return None
    
    # Calculate padding to center the QR code
    left_padding = (target_size[0] - img.size[0]) // 2
    top_padding = (target_size[1] - img.size[1]) // 2
    
    # Create a new white image of target size and paste the QR code in the center
    new_img = Image.new('L', target_size, color=255)  # Use 'L' mode for grayscale
    new_img.paste(img.convert('L'), (left_padding, top_padding))
    
    # Save the image
    new_img.save(file_path)
    logger.debug(f"QR code saved to {file_path} with size {new_img.size}")
    
    # Log the final box_size and border used
    logger.debug(f"Final box_size: {box_size}, border: {border}")
    if box_size != config['box_size'] or border != config['border']:
        logger.warning(f"QR code parameters adjusted from config values. Original: box_size={config['box_size']}, border={config['border']}")
    
    # Preprocess the image
    img_array = np.array(new_img)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.where(img_array > 0.5, 1.0, 0.0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array, new_img.size