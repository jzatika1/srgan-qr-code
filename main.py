import argparse
import yaml
import logging.config
import os
from src.qr_code_generator import generate_qr_code
from src.dataset_preparation import create_dataset
from src.srgan_trainer import train_srgan
from src.utils import ensure_directories_exist

def parse_args():
    parser = argparse.ArgumentParser(description="Modular SRGAN for QR Code Generation")
    parser.add_argument('--generate_qr', action='store_true', help='Generate a new QR code, overwriting existing one if present')
    parser.add_argument('--create_dataset', action='store_true', help='Create dataset with QR code variations')
    parser.add_argument('--train_srgan', action='store_true', help='Start SRGAN training')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file')
    parser.add_argument('--log_config', type=str, default='config/logger.yaml', help='Path to the logger config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(log_config_path):
    with open(log_config_path, 'r') as file:
        log_config = yaml.safe_load(file)
        # Disable PIL debug logging
        logging.getLogger('PIL').setLevel(logging.INFO)
        logging.config.dictConfig(log_config)
    return logging.getLogger(__name__)

def ensure_all_directories(config):
    directories = [
        os.path.dirname(config['qr_code']['perfect_qr_dir']),
        config['dataset']['output_dir'],
        config['srgan']['data']['dataset_dir'],
        config['srgan']['data']['generated_dir']
    ]
    ensure_directories_exist(directories)

def load_or_generate_qr(config, logger, force_generate=False):
    qr_config = config['qr_code']
    qr_config['force_generate'] = force_generate
    perfect_qr, qr_size = generate_qr_code(qr_config)
    if perfect_qr is None:
        logger.error("Failed to generate or load QR code.")
        return None, None
    logger.info(f"QR code size: {qr_size}")
    return perfect_qr, qr_size

def create_dataset_wrapper(config, logger, perfect_qr):
    try:
        config['dataset']['perfect_qr'] = perfect_qr
        train_dir, val_dir, dataset_info_path = create_dataset(config['dataset'])
        logger.info(f"Dataset created successfully. Train dir: {train_dir}, Val dir: {val_dir}")
        logger.info(f"Dataset info saved to: {dataset_info_path}")
        
        config['srgan']['data']['train_dir'] = train_dir
        config['srgan']['data']['val_dir'] = val_dir
        config['srgan']['data']['dataset_info'] = dataset_info_path
        return config
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        return None

def train_srgan_wrapper(config, logger):
    if 'train_dir' not in config['srgan']['data'] or 'val_dir' not in config['srgan']['data']:
        logger.error("Training and validation directories not set. Please run --create_dataset first.")
        return False
    try:
        train_srgan(config['srgan'], config['dataset'])
        return True
    except Exception as e:
        logger.error(f"Error during SRGAN training: {str(e)}")
        return False

def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(args.log_config)
    
    ensure_all_directories(config)
    
    perfect_qr, qr_size = load_or_generate_qr(config, logger, force_generate=args.generate_qr)
    if perfect_qr is None:
        return
    
    if args.create_dataset:
        config = create_dataset_wrapper(config, logger, perfect_qr)
        if config is None:
            return
    
    if args.train_srgan:
        success = train_srgan_wrapper(config, logger)
        if not success:
            return

if __name__ == '__main__':
    main()