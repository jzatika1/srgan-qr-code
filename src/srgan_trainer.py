import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, PReLU, BatchNormalization, Add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from PIL import Image
from src.utils import get_function_logger
from src.srgan_model import build_discriminator, build_vgg
from tqdm import tqdm
import matplotlib.pyplot as plt

def build_generator(config):
    def residual_block(x, filters, kernel_size):
        res = Conv2D(filters, kernel_size, padding='same')(x)
        res = BatchNormalization()(res)
        res = PReLU(shared_axes=[1, 2])(res)
        res = Conv2D(filters, kernel_size, padding='same')(res)
        res = BatchNormalization()(res)
        return Add()([x, res])

    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(config['initial_filters'], (9, 9), padding='same')(input_layer)
    x = PReLU(shared_axes=[1, 2])(x)

    for _ in range(config['residual_blocks']):
        x = residual_block(x, config['initial_filters'], config['kernel_size'])

    x = Conv2D(config['initial_filters'], config['kernel_size'], padding='same')(x)
    x = BatchNormalization()(x)

    # Upsampling layers to increase resolution
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
    x = PReLU(shared_axes=[1, 2])(x)

    output_layer = Conv2D(3, (9, 9), padding='same', activation='tanh')(x)

    return Model(inputs=input_layer, outputs=output_layer)


def load_dataset(srgan_config, dataset_config, batch_size, hr_size=(128, 128), lr_size=(32, 32)):
    logger = get_function_logger()
    logger.info("Starting dataset loading process")
    
    def preprocess(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)  # Ensure 3 channels
        image = tf.image.resize(image, lr_size)
        image = (tf.cast(image, tf.float32) / 127.5) - 1.0
        logger.debug(f"Preprocessed image shape: {image.shape}")
        return image

    # Load the single HR image
    hr_image_path = dataset_config['hq_img_path']
    logger.info(f"Loading HR image from: {hr_image_path}")
    if not os.path.exists(hr_image_path):
        logger.error(f"HR image not found at {hr_image_path}")
        raise FileNotFoundError(f"HR image not found at {hr_image_path}")

    hr_image = tf.io.read_file(hr_image_path)
    hr_image = tf.image.decode_png(hr_image, channels=3)  # Ensure 3 channels
    hr_image = tf.image.resize(hr_image, hr_size)
    hr_image = (tf.cast(hr_image, tf.float32) / 127.5) - 1.0
    hr_image = tf.expand_dims(hr_image, 0)
    logger.info(f"HR image loaded and processed. Shape: {hr_image.shape}")

    # Load LR images
    lr_dir = srgan_config['data']['dataset_dir']
    logger.info(f"Loading LR images from: {lr_dir}")
    if not os.path.exists(lr_dir):
        logger.error(f"LR image directory not found at {lr_dir}")
        raise FileNotFoundError(f"LR image directory not found at {lr_dir}")

    # Get all image files
    all_images = [os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith('.png')]
    logger.info(f"Found {len(all_images)} image files")

    # Load validation samples
    val_samples_file = os.path.join(lr_dir, 'val_samples.txt')
    if os.path.exists(val_samples_file):
        with open(val_samples_file, 'r') as f:
            val_samples = set(f.read().splitlines())
        train_images = [img for img in all_images if os.path.basename(img) not in val_samples]
        val_images = [img for img in all_images if os.path.basename(img) in val_samples]
    else:
        logger.warning("val_samples.txt not found. Using random split.")
        val_split = dataset_config['val_split']
        num_val = int(len(all_images) * val_split)
        train_images = all_images[:-num_val]
        val_images = all_images[-num_val:]

    logger.info(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_images)).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_images)
    val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size)

    # Add the HR image to each batch
    def add_hr_image(lr_batch):
        batch_size = tf.shape(lr_batch)[0]
        hr_batch = tf.repeat(hr_image, repeats=batch_size, axis=0)
        logger.debug(f"LR batch shape: {lr_batch.shape}, HR batch shape: {hr_batch.shape}")
        return lr_batch, hr_batch

    train_dataset = train_dataset.map(add_hr_image)
    val_dataset = val_dataset.map(add_hr_image)

    logger.info(f"Train dataset size: {tf.data.experimental.cardinality(train_dataset)}")
    logger.info(f"Validation dataset size: {tf.data.experimental.cardinality(val_dataset)}")

    logger.info("Dataset loading completed successfully")
    return train_dataset, val_dataset

def train_srgan(srgan_config, dataset_config):
    logger = get_function_logger()
    logger.info("=== Starting SRGAN Training ===")
    
    logger.info("Configuration:")
    logger.info(f"Generator: {srgan_config['generator']}")
    logger.info(f"Discriminator: {srgan_config['discriminator']}")
    logger.info(f"Training: {srgan_config['training']}")
    
    logger.info("Building models...")
    generator = build_generator(srgan_config['generator'])
    discriminator = build_discriminator(srgan_config['discriminator'])
    logger.info("Models built successfully.")
    
    logger.info("Loading dataset...")
    try:
        train_dataset, val_dataset = load_dataset(srgan_config, dataset_config, srgan_config['training']['batch_size'])
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return
    
    logger.info(f"Train dataset size: {tf.data.experimental.cardinality(train_dataset)}")
    logger.info(f"Validation dataset size: {tf.data.experimental.cardinality(val_dataset)}")
    
    logger.info("Building VGG model for perceptual loss...")
    vgg = build_vgg((128, 128, 3))
    vgg.trainable = False
    
    logger.info("Compiling models...")
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=srgan_config['training']['generator_learning_rate'], 
                                                   beta_1=srgan_config['training']['beta1'], 
                                                   beta_2=srgan_config['training']['beta2'])
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=srgan_config['training']['discriminator_learning_rate'], 
                                                       beta_1=srgan_config['training']['beta1'], 
                                                       beta_2=srgan_config['training']['beta2'])
    
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(lr_imgs, hr_imgs):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_hr_imgs = generator(lr_imgs, training=True)
            
            # Ensure hr_imgs has the correct shape
            if hr_imgs.shape.ndims == 3:
                hr_imgs = tf.expand_dims(hr_imgs, axis=-1)
            
            # Ensure fake_hr_imgs has the correct shape
            if fake_hr_imgs.shape.ndims == 3:
                fake_hr_imgs = tf.expand_dims(fake_hr_imgs, axis=-1)
            
            noise_real = tf.random.normal(shape=hr_imgs.shape, mean=0.0, stddev=0.1)
            noise_fake = tf.random.normal(shape=fake_hr_imgs.shape, mean=0.0, stddev=0.1)
            
            real_output = discriminator(hr_imgs + noise_real, training=True)
            fake_output = discriminator(fake_hr_imgs + noise_fake, training=True)
            
            content_loss = mse(vgg(hr_imgs), vgg(fake_hr_imgs))
            
            # Adversarial loss (least squares)
            adversarial_loss = tf.reduce_mean(tf.square(fake_output - 1))
        
            gen_loss = content_loss + srgan_config['training']['lambda'] * adversarial_loss
            
            real_loss = bce(tf.random.uniform(real_output.shape, 0.8, 1.0), real_output)
            fake_loss = bce(tf.random.uniform(fake_output.shape, 0.0, 0.2), fake_output)
            disc_loss = real_loss + fake_loss
        
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        return gen_loss, disc_loss
    
    logger.info("Ensuring output directories exist...")
    os.makedirs(srgan_config['data']['generated_dir'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    best_gen_loss = float('inf')
    patience = 10
    no_improvement = 0
    
    gen_losses = []
    disc_losses = []
    
    logger.info("=== Starting Training Loop ===")
    try:
        for epoch in range(srgan_config['training']['epochs']):
            logger.info(f"Epoch {epoch+1}/{srgan_config['training']['epochs']}")
            pbar = tqdm(enumerate(train_dataset), total=tf.data.experimental.cardinality(train_dataset).numpy())
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            steps = 0
            
            logger.info("Iterating over batches...")
            for step, (lr_imgs, hr_imgs) in pbar:
                logger.debug(f"Processing batch {step+1}")
                try:
                    gen_loss, disc_loss = train_step(lr_imgs, hr_imgs)
                    epoch_gen_loss += gen_loss
                    epoch_disc_loss += disc_loss
                    steps += 1
                    pbar.set_description(f"Epoch {epoch+1}/{srgan_config['training']['epochs']}")
                    pbar.set_postfix({'gen_loss': f'{gen_loss:.4f}', 'disc_loss': f'{disc_loss:.4f}'})
                except tf.errors.OutOfRangeError:
                    logger.warning(f"End of sequence reached at step {step+1}. Moving to next epoch.")
                    break
                except Exception as e:
                    logger.error(f"Error during training step {step+1}: {str(e)}")
                    raise
            
            if steps == 0:
                logger.error("No steps were performed in this epoch. The dataset might be empty.")
                break
            
            avg_gen_loss = epoch_gen_loss / steps
            avg_disc_loss = epoch_disc_loss / steps
            gen_losses.append(avg_gen_loss)
            disc_losses.append(avg_disc_loss)
            
            logger.info(f"Epoch {epoch+1} Summary:")
            logger.info(f"  Steps completed: {steps}")
            logger.info(f"  Generator Loss: {avg_gen_loss:.4f}")
            logger.info(f"  Discriminator Loss: {avg_disc_loss:.4f}")
            
            if avg_gen_loss < best_gen_loss:
                best_gen_loss = avg_gen_loss
                no_improvement = 0
                logger.info("  New best generator loss achieved.")
            else:
                no_improvement += 1
                logger.info(f"  No improvement for {no_improvement} epochs.")
            
            if no_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            if (epoch + 1) % srgan_config['training']['save_interval'] == 0:
                generator.save(f'models/generator_epoch_{epoch+1}.keras')
                discriminator.save(f'models/discriminator_epoch_{epoch+1}.keras')
                logger.info(f"  Saved models for epoch {epoch+1}")
            
            logger.info("  Generating and saving sample images...")
            for lr_imgs, hr_imgs in val_dataset.take(1):
                generated_images = generator(lr_imgs, training=False)
                for i, img in enumerate(generated_images):
                    img = (img * 127.5 + 127.5).numpy().astype(np.uint8)
                    img = Image.fromarray(img)
                    save_path = os.path.join(srgan_config['data']['generated_dir'], f"epoch_{epoch+1}_img_{i}.png")
                    img.save(save_path)
            logger.info("  Sample images saved successfully.")
    
        logger.info("=== SRGAN Training Completed Successfully ===")
        
        if gen_losses and disc_losses:
            logger.info("Plotting and saving loss history...")
            plt.figure(figsize=(10, 5))
            plt.plot(gen_losses, label='Generator Loss')
            plt.plot(disc_losses, label='Discriminator Loss')
            plt.legend()
            plt.savefig('loss_history.png')
            logger.info("Loss history plot saved to loss_history.png")
        else:
            logger.warning("No loss history to plot. Training might not have occurred.")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

    logger.info("Training process finished.")