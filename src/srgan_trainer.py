import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, PReLU, BatchNormalization, Add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from PIL import Image
from src.utils import get_function_logger
from src.srgan_model import build_discriminator
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

    input_layer = Input(shape=(128, 128, 1))
    x = Conv2D(config['initial_filters'], (9, 9), padding='same')(input_layer)
    x = PReLU(shared_axes=[1, 2])(x)

    for _ in range(config['residual_blocks']):
        x = residual_block(x, config['initial_filters'], config['kernel_size'])

    x = Conv2D(config['initial_filters'], config['kernel_size'], padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = PReLU(shared_axes=[1, 2])(x)

    output_layer = Conv2D(1, (9, 9), padding='same')(x)
    output_layer = Lambda(lambda x: tf.clip_by_value(x, 0, 1))(output_layer)

    return Model(inputs=input_layer, outputs=output_layer)

def load_dataset(srgan_config, dataset_config, batch_size):
    logger = get_function_logger()
    logger.info("Starting dataset loading process")
    
    def preprocess(lr_path):
        lr_image = tf.io.read_file(lr_path)
        lr_image = tf.image.decode_png(lr_image, channels=1)
        lr_image = tf.cast(lr_image, tf.float32) / 255.0
        lr_image = tf.where(lr_image > 0.5, 1.0, 0.0)  # Thresholding
        return lr_image

    # Load the single HR image
    hr_image_path = dataset_config['hq_img_path']
    logger.info(f"Loading HR image from: {hr_image_path}")
    if not os.path.exists(hr_image_path):
        logger.error(f"HR image not found at {hr_image_path}")
        raise FileNotFoundError(f"HR image not found at {hr_image_path}")

    hr_image = tf.io.read_file(hr_image_path)
    hr_image = tf.image.decode_png(hr_image, channels=1)
    hr_image = tf.cast(hr_image, tf.float32) / 255.0
    hr_image = tf.where(hr_image > 0.5, 1.0, 0.0)  # Thresholding
    hr_image = tf.expand_dims(hr_image, 0)  # Add batch dimension
    logger.info(f"HR image loaded and processed. Shape: {hr_image.shape}")

    train_dir = srgan_config['data']['train_dir']
    val_dir = srgan_config['data']['val_dir']

    # Get all LR images
    train_lr_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.png')]
    val_lr_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.png')]

    logger.info(f"Training images: {len(train_lr_images)}, Validation images: {len(val_lr_images)}")

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_lr_images)
    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1000)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_lr_images)
    val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Add the HR image to each sample
    def add_hr_image(lr_image):
        return lr_image, hr_image[0]

    train_dataset = train_dataset.map(add_hr_image)
    val_dataset = val_dataset.map(add_hr_image)

    # Log the number of images before batching
    logger.info(f"Number of training images: {len(train_lr_images)}")
    logger.info(f"Number of validation images: {len(val_lr_images)}")

    # Now batch the datasets
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    # Log the number of batches
    train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    val_batches = tf.data.experimental.cardinality(val_dataset).numpy()
    logger.info(f"Number of training batches: {train_batches}")
    logger.info(f"Number of validation batches: {val_batches}")

    logger.info("Dataset loading completed successfully")
    return train_dataset, val_dataset

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def least_squares_loss(y_true, y_pred):
    return 0.5 * tf.reduce_mean(tf.square(y_true - y_pred))

def gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = tf.random.uniform([real_samples.shape[0], 1, 1, 1], 0.0, 1.0)
    interpolated = real_samples + alpha * (fake_samples - real_samples)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def train_srgan(srgan_config, dataset_config):
    logger = get_function_logger()
    logger.info("=== Starting SRGAN Training for QR Codes ===")
    
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
    
    logger.info("Compiling models...")
    generator_optimizer = Adam(learning_rate=srgan_config['training']['generator_learning_rate'], 
                               beta_1=srgan_config['training']['beta1'], 
                               beta_2=srgan_config['training']['beta2'])
    discriminator_optimizer = Adam(learning_rate=srgan_config['training']['discriminator_learning_rate'], 
                                   beta_1=srgan_config['training']['beta1'], 
                                   beta_2=srgan_config['training']['beta2'])
    
    mse = tf.keras.losses.MeanSquaredError()
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(lr_imgs, hr_imgs):
        def normalize_images(images):
            return tf.clip_by_value(images, 0, 1)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_hr_imgs = generator(lr_imgs, training=True)
            
            real_output = discriminator(normalize_images(hr_imgs), training=True)
            fake_output = discriminator(normalize_images(fake_hr_imgs), training=True)
            
            # Content loss (MSE)
            content_loss = mse(hr_imgs, fake_hr_imgs)
            
            # Binary cross-entropy loss
            bce_loss = binary_crossentropy(hr_imgs, fake_hr_imgs)
            
            # Adversarial loss
            gen_loss = binary_crossentropy(tf.ones_like(fake_output), fake_output)
            
            # Total generator loss
            total_gen_loss = (
                srgan_config['training']['content_loss_weight'] * content_loss + 
                srgan_config['training']['bce_loss_weight'] * bce_loss +
                srgan_config['training']['adversarial_loss_weight'] * gen_loss
            )
            
            # Discriminator loss
            real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
            fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
            total_disc_loss = real_loss + fake_loss
        
        # Compute gradients
        gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
        
        # Apply gradients
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        return total_gen_loss, total_disc_loss
    
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
            for lr_imgs, _ in val_dataset.take(1):
                generated_images = generator(lr_imgs, training=False)
                for i, gen_img in enumerate(generated_images):
                    # Process generated image
                    gen_img = tf.clip_by_value(gen_img, -1, 1)
                    gen_img = ((gen_img + 1) * 127.5).numpy().astype(np.uint8)
                    gen_img = np.squeeze(gen_img)  # Remove the channel dimension
                    gen_img = Image.fromarray(gen_img, mode='L')  # 'L' mode for grayscale
                
                    # Save generated image
                    gen_save_path = os.path.join(srgan_config['data']['generated_dir'], f"epoch_{epoch+1}_gen_img_{i}.png")
                    gen_img.save(gen_save_path)
                
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