import argparse
import yaml
import logging.config
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
        config['qr_code']['file_path'],
        config['dataset']['output_dir'],
        config['srgan']['data']['dataset_dir'],
        config['srgan']['data']['generated_dir']
    ])
    
    if args.generate_qr:
        generate_qr_code(config['qr_code'])
    
    if args.create_dataset:
        create_dataset(config['dataset'])
    
    if args.train_srgan:
        train_srgan(config['srgan'], config['dataset'])

if __name__ == '__main__':
    main()
