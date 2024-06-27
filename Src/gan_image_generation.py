import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt

# Set seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define hyperparameters
BUFFER_SIZE = 60000
BATCH_SIZE = 128
NOISE_DIM = 100
EPOCHS = 50
GENERATED_IMAGES_DIR = "C:\\Users\........."

# Define file path for dataset
dataset_path = "C:\\Users\............."

# Load dataset
def load_dataset():
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        image_size=(416, 416),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )
    # Normalize images
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 127.5) - 1,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    # Prefetch for performance
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_dataset

# Create discriminator
def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[416, 416, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def make_generator_model():
    model = keras.Sequential()

    model.add(layers.Dense(13 * 13 * 256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((13, 13, 256)))
    assert model.output_shape == (None, 13, 13, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 13, 13, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 26, 26, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 52, 52, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 416, 416, 1)

    return model



# Define the GAN model
def make_gan_model(generator, discriminator):
    discriminator.trainable = False
    gan_input = keras.Input(shape=(NOISE_DIM,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-4))
    return gan


# Define training loop
def train_gan(gan, dataset):
    # Create generator input for viewing progress during training
    noise_for_preview = tf.random.normal([10, NOISE_DIM])

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for i, real_images in enumerate(dataset):
            # Generate fake images
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            fake_images = gan.generator(noise)

            # Concatenate real and fake images for discriminator input
            discriminator_input = tf.concat([real_images, fake_images], axis=0)

            # Create target labels for discriminator training
            target_labels = tf.concat([tf.ones((BATCH_SIZE, 1)), tf.zeros((BATCH_SIZE, 1))], axis=0)

            # Add noise to target labels to improve GAN training
            target_labels += 0.05 * tf.random.uniform(tf.shape(target_labels))

            # Train discriminator
            discriminator_loss = gan.discriminator.train_on_batch(discriminator_input, target_labels)

            # Generate noise for generator input and set target labels to ones
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            target_labels = tf.ones((BATCH_SIZE, 1))

            # Train generator via GAN model
            gan_loss = gan.train_on_batch(noise, target_labels)

            # Output progress during training
            if i % 100 == 0:
                # Preview progress by generating images from fixed noise input
                generated_images = gan.generator(noise_for_preview)
                generated_images = 0.5 * generated_images + 0.5  # Rescale images to [0, 1]
                fig, axs = plt.subplots(1, 10, figsize=(20, 2))
                for j in range(10):
                    axs[j].imshow(generated_images[j], cmap='gray')
                    axs[j].axis('off')
                plt.show()

            print(f"Batch {i+1}/{BUFFER_SIZE//BATCH_SIZE} - discriminator loss: {discriminator_loss:.4f}, generator loss: {gan_loss:.4f}")

# Load dataset
train_dataset = load_dataset()

# Create discriminator and generator
discriminator = make_discriminator_model()
generator = make_generator_model()

# Create GAN model
gan = make_gan_model(generator, discriminator)

# Train GAN model
train_gan(gan, train_dataset)
# Define functions to generate and save images
GENERATED_IMAGES_DIR = 'generated_images'

# Save generated images for visualization during training
preview_images = gan.generator(noise_for_preview, training=False)
fig, axs = plt.subplots(1, 10, figsize=(20, 2))
for j in range(10):
    axs[j].imshow(preview_images[j].numpy().reshape(416, 416), cmap='gray')
    axs[j].axis('off')
plt.show()
plt.close()

# Save generated images every 5 epochs
if (epoch + 1) % 5 == 0 and i == 0:
    save_generated_images(gan.generator, epoch + 1)

# Output loss
if i % 10 == 0:
    print(
        f"\tBatch {i + 1}/{BUFFER_SIZE // BATCH_SIZE} - Generator Loss: {gan_loss:.4f}, Discriminator Loss: {discriminator_loss:.4f}")

# Define function to save generated images
def save_generated_images(generator, epoch):
    os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)


noise = tf.random.normal([100, NOISE_DIM])
generated_images = generator(noise, training=False)
for i in range(generated_images.shape[0]):
    plt.imshow(generated_images[i].numpy().reshape(416, 416), cmap='gray')
    plt.axis('off')
    plt.savefig(f"{GENERATED_IMAGES_DIR}/{epoch}epoch{i}_generated.png")
    plt.close()


# Load the dataset
train_dataset = load_dataset()

# Create the models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Create the GAN model
gan = make_gan_model(generator, discriminator)

# Train the GAN model
train_gan(gan, train_dataset)

# Define function to save generated images
def save_generated_images(generated_images, epoch):
    fig = plt.figure(figsize=(4, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((generated_images[i, :, :, 0] + 1) / 2, cmap='gray')
        plt.axis('off')

    # Save generated images to disk
    if not os.path.exists(GENERATED_IMAGES_DIR):
        os.makedirs(GENERATED_IMAGES_DIR)
    plt.savefig(os.path.join(GENERATED_IMAGES_DIR, f"generated_images_{epoch+1}.png"))
    plt.close()

