# Modular SRGAN for QR Code Generation

This project implements a Super-Resolution Generative Adversarial Network (SRGAN) to generate high-resolution QR codes from low-resolution images. The project is modular and includes functionalities for generating QR codes, creating datasets, and training the SRGAN model.

## Usage

To use the script, run `main.py` with the appropriate options. The available options are:

```
usage: main.py [-h] [--generate_qr] [--create_dataset] [--train_srgan] [--config CONFIG] [--log_config LOG_CONFIG]

options:
  -h, --help            show this help message and exit
  --generate_qr         Generate the default QR code
  --create_dataset      Create dataset with QR code variations
  --train_srgan         Start SRGAN training
  --config CONFIG       Path to the config file
  --log_config LOG_CONFIG
                        Path to the logger config file
```

## Options

- `-h, --help`: Show the help message and exit.
- `--generate_qr`: Generate the default QR code.
- `--create_dataset`: Create a dataset with QR code variations.
- `--train_srgan`: Start training the SRGAN model.
- `--config CONFIG`: Specify the path to the configuration file.
- `--log_config LOG_CONFIG`: Specify the path to the logger configuration file.

## Examples

### Generate a QR Code

```
python main.py --generate_qr
```

### Create a Dataset

```
python main.py --create_dataset --config path/to/config.yaml
```

### Train the SRGAN Model

```
python main.py --train_srgan --config path/to/config.yaml --log_config path/to/log_config.yaml
```

## Project Structure

- `main.py`: The main script to run the different functionalities.
- `src/`: Source code directory containing the implementation of the SRGAN model, dataset creation, and utility functions.
- `data/`: Directory to store datasets (ignored by Git).
- `models/`: Directory to save trained models.
- `logs/`: Directory to save logs (if specified in the logger configuration).

## Configuration

Configuration files are used to specify parameters for QR code generation, dataset creation, and SRGAN training. Example configuration files are provided in the `configs/` directory.

## Logging

Logging configurations can be specified using a logging configuration file. Example logging configuration files are provided in the `configs/` directory.

## Requirements

Ensure you have the required packages installed using `conda`. You can create the environment using the provided `environment.yml` file:

```
name: srgan-qr-code
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - numpy
  - pandas
  - scikit-learn
  - pillow
  - matplotlib
  - pyyaml
  - jupyterlab
  - ipykernel
  - pip=24.0
  - pip:
      - qrcode[pil]
      - tensorflow[and-cuda]
      - tqdm
```

To create the environment, run:

```
conda env create -f environment.yml
```