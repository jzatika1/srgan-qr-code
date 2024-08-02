from tensorflow.keras.layers import Input, Conv2D, PReLU, BatchNormalization, Add, LeakyReLU, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from typing import Dict
from src.utils import get_function_logger

def build_generator(config: Dict) -> Model:
    """
    Build the generator model for SRGAN.
    
    Args:
    config (Dict): Configuration dictionary for the generator.
    
    Returns:
    Model: Keras Model instance of the generator.
    """
    def residual_block(x, filters: int, kernel_size: int):
        res = Conv2D(filters, kernel_size, padding='same')(x)
        res = BatchNormalization()(res)
        res = PReLU(shared_axes=[1, 2])(res)
        res = Conv2D(filters, kernel_size, padding='same')(res)
        res = BatchNormalization()(res)
        return Add()([x, res])

    input_layer = Input(shape=(128, 128, 1))
    x = Conv2D(config['initial_filters'], (9, 9), padding='same')(input_layer)
    x = PReLU(shared_axes=[1, 2])(x)

    for _ in range(config['residual_blocks']):
        x = residual_block(x, config['initial_filters'], config['kernel_size'])

    x = Conv2D(config['initial_filters'], config['kernel_size'], padding='same')(x)
    x = BatchNormalization()(x)
    output_layer = Conv2D(1, (9, 9), padding='same', activation='tanh')(x)

    return Model(inputs=input_layer, outputs=output_layer)

def build_discriminator(config: Dict) -> Model:
    input_layer = Input(shape=(128, 128, 1))
    x = Conv2D(config['initial_filters'], (config['kernel_size'], config['kernel_size']), padding='same')(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    filters = config['initial_filters']
    for _ in range(3):  # Reduced from 4 to 3 layers
        filters *= 2
        x = Conv2D(filters, (config['kernel_size'], config['kernel_size']), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.2)(x)
    x = Flatten()(x)
    x = Dense(512)(x)  # Reduced from 1024 to 512
    x = LeakyReLU(negative_slope=0.2)(x)
    output_layer = Dense(1)(x)
    return Model(inputs=input_layer, outputs=output_layer)

"""
def build_vgg(config: Dict) -> Model:
    #Build a VGG model for perceptual loss, adapting it for grayscale input.
    
    #Args:
    #config (Dict): Configuration dictionary for the VGG model.
    
    #Returns:
    #Model: Keras Model instance of the VGG for perceptual loss.
    logger = get_function_logger()
    
    # Create a grayscale input
    grayscale_input = Input(shape=(128, 128, 1))
    logger.info(f"VGG grayscale input shape: {grayscale_input.shape}")
    
    # Convert grayscale to RGB by repeating the channel
    rgb_input = Lambda(lambda x: tf.repeat(x, 3, axis=-1))(grayscale_input)
    logger.info(f"VGG rgb input shape: {rgb_input.shape}")
    
    # Load pre-trained VGG19 model
    vgg = VGG19(weights=config['weights'], include_top=False, input_shape=(128, 128, 3))
    
    # Create a new model that takes grayscale input
    vgg_model = Model(inputs=grayscale_input, outputs=vgg(rgb_input))
    
    # Get the output of the specified layer
    target_layer = config.get('layer', 'vgg19')
    
    if isinstance(target_layer, int):
        if 0 <= target_layer < len(vgg.layers):
            target_layer = vgg.layers[target_layer].name
        else:
            logger.warning(f"Layer index {target_layer} is out of range. Using 'vgg19' instead.")
            target_layer = 'vgg19'
    elif isinstance(target_layer, str):
        if target_layer not in [layer.name for layer in vgg.layers]:
            logger.warning(f"Layer '{target_layer}' not found in VGG19 model. Using 'vgg19' instead.")
            target_layer = 'vgg19'
    else:
        logger.warning(f"Invalid layer specification: {target_layer}. Using 'vgg19' instead.")
        target_layer = 'vgg19'
    
    vgg_output = vgg_model.get_layer(target_layer).output
    logger.info(f"VGG output shape from layer '{target_layer}': {vgg_output.shape}")
    
    # Create and return the final model
    return Model(inputs=grayscale_input, outputs=vgg_output)
    """
