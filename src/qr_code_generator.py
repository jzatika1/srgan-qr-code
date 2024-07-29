import qrcode
import os
from src.utils import get_function_logger, ensure_directories_exist

def generate_qr_code(config):
    logger = get_function_logger()
    logger.debug("Starting QR code generation.")
    
    file_path = config['file_path']
    ensure_directories_exist([file_path])
    
    message = config['message']
    qr = qrcode.QRCode(
        version=config['version'],
        error_correction=getattr(qrcode.constants, f'ERROR_CORRECT_{config["error_correction"]}'),
        box_size=config['box_size'],
        border=config['border']
    )
    qr.add_data(message)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.save(file_path)
    logger.debug(f"QR code saved to {file_path}")
