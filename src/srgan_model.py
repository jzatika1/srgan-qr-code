from tensorflow.keras.layers import Input, Conv2D, PReLU, BatchNormalization, Add, LeakyReLU, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

def build_generator(config):
    def residual_block(x, filters, kernel_size):
        res = Conv2D(filters, kernel_size, padding='same')(x)
        res = BatchNormalization()(res)
        res = PReLU(shared_axes=[1, 2])(res)  # Apply shared_axes
        res = Conv2D(filters, kernel_size, padding='same')(res)
        res = BatchNormalization()(res)
        return Add()([x, res])

    input_layer = Input(shape=(128, 128, 3))  # Define input shape

    x = Conv2D(config['initial_filters'], (9, 9), padding='same')(input_layer)
    x = PReLU(shared_axes=[1, 2])(x)  # Apply shared_axes

    for _ in range(config['residual_blocks']):
        x = residual_block(x, config['initial_filters'], config['kernel_size'])

    x = Conv2D(config['initial_filters'], config['kernel_size'], padding='same')(x)
    x = BatchNormalization()(x)
    output_layer = Conv2D(3, (9, 9), padding='same', activation='tanh')(x)

    return Model(inputs=input_layer, outputs=output_layer)

def build_discriminator(config):
    input_layer = Input(shape=(128, 128, 3))  # Define input shape
    x = Conv2D(config['initial_filters'], (config['kernel_size'], config['kernel_size']), padding='same')(input_layer)
    x = LeakyReLU(alpha=0.2)(x)

    for _ in range(3):
        x = Conv2D(config['initial_filters'] * 2, (config['kernel_size'], config['kernel_size']), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        config['initial_filters'] *= 2

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    return Model(inputs=input_layer, outputs=output_layer)

def build_vgg(hr_shape):
    """Build a VGG model for perceptual loss."""
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)
