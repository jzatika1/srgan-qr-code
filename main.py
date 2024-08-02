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
    parser.add_argument('--generate_qr', action='store_true', help='Generate the default QR code')
    parser.add_argument('--create_dataset', action='store_true', help='Create dataset with QR code variations')
    parser.add_argument('--train_srgan', action='store_true', help='Start SRGAN training')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file')
    parser.add_argument('--log_config', type=str, default='config/logger.yaml', help='Path to the logger config file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    with open(args.log_config, 'r') as file:
        log_config = yaml.safe_load(file)
        logging.config.dictConfig(log_config)
        
    logger = logging.getLogger(__name__)
    
    # Ensure necessary directories exist
    ensure_directories_exist([
        os.path.dirname(config['qr_code']['perfect_qr_dir']),
        config['dataset']['output_dir'],
        config['srgan']['data']['dataset_dir'],
        config['srgan']['data']['generated_dir']
    ])
    
    # Always generate the QR code if it doesn't exist or if explicitly requested
    if args.generate_qr or 'perfect_qr' not in config['dataset']:
        perfect_qr, qr_size = generate_qr_code(config['qr_code'])
        logger.info(f"Generated QR code size: {qr_size}")
        if qr_size != (128, 128):
            logger.warning(f"Generated QR code size {qr_size} is not 128x128. This may cause issues in training.")
        
        # Save the preprocessed perfect QR code to the config for later use
        config['dataset']['perfect_qr'] = perfect_qr
    
    if args.create_dataset:
        try:
            train_dir, val_dir, dataset_info_path = create_dataset(config['dataset'])
            logger.info(f"Dataset created successfully. Train dir: {train_dir}, Val dir: {val_dir}")
            logger.info(f"Dataset info saved to: {dataset_info_path}")
            
            # Update config with new paths
            config['srgan']['data']['train_dir'] = train_dir
            config['srgan']['data']['val_dir'] = val_dir
            config['srgan']['data']['dataset_info'] = dataset_info_path
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            return
    
    if args.train_srgan:
        if 'train_dir' not in config['srgan']['data'] or 'val_dir' not in config['srgan']['data']:
            logger.error("Training and validation directories not set. Please run --create_dataset first.")
            return
        try:
            train_srgan(config['srgan'], config['dataset'])
        except Exception as e:
            logger.error(f"Error during SRGAN training: {str(e)}")
            return

if __name__ == '__main__':
    main()