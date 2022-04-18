#---The GAN will be trained with Pix2Pix method---#
#---The input images are of size 128x128, grey----#

import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
import datetime

# Make the GPU(MX150) compatible
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [128, 256])

    w = tf.shape(image)[1]
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 140, 140)
    input_image, real_image = random_crop(input_image, real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def random_input():
    rand_input = tf.keras.Sequential()
    rand_input.add(tf.keras.layers.Dense(16 * 16 * 512, use_bias=False, input_shape=(100,)))
    rand_input.add(tf.keras.layers.BatchNormalization())
    rand_input.add(tf.keras.layers.LeakyReLU())

    rand_input.add(tf.keras.layers.Reshape((16, 16, 512)))  # 修改图片大小
    assert rand_input.output_shape == (None, 16, 16, 512)  # batch size 没有限制

    rand_input.add(tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2),
                                                   padding='same', use_bias=False))  # 反卷积
    assert rand_input.output_shape == (None, 32, 32, 256)

    return rand_input


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def downsample_same(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def upsample_same(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=1,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def make_generator():
    a = tf.keras.layers.Input(shape=[128, 128, 1])
    b = tf.keras.layers.Input(shape=[32, 32, 256])
    inputs = [a, b]

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 64, 64, 64)
        downsample_same(128, 4),    # (bs, 64, 64, 128)
        downsample_same(256, 4),    # (bs, 64, 64, 256)
        downsample(256, 4),  # (bs, 32, 32, 512), rdm will be concatenated here
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]
    # up_stack will be concatenated with the layers in down_stack
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(512, 4),  # (bs, 32, 32, 512)
        upsample(256, 4),  # (bs, 64, 64, 256)
        upsample_same(128, 4),  # (bs, 64, 64, 128)
        upsample_same(64, 4),  # (bs, 64, 64, 64)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 128, 128, 1)

    x = a
    # Insert random noise
    skips = []
    k = 0
    for down in down_stack:
        k += 1
        x = down(x)
        if k == 4:
            x = tf.keras.layers.Concatenate(axis=-1)([x, b])
        skips.append(x)

    skips = reversed(skips[:-1])  # 列表skips从第0位到最后一位前的一位（不含最后一位） 取倒序

    # upsampling and building the connection
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def make_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[128, 128, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[128, 128, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 128, 128, channels*2)

    down1 = downsample(32, 4, False)(x)  # (bs, 64, 64, 32)
    down2 = downsample(64, 4)(down1)  # (bs, 32, 32, 64)
    down3 = downsample_same(128, 4)(down2)  # (bs, 32, 32, 128)
    down4 = downsample_same(256, 4)(down3)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    t_i_0 = test_input[0]
    t_i = t_i_0[0]
    ta = tar[0]
    pre = prediction[0]
    display_list = [t_i[:, :, 0], ta[:, :, 0], pre[:, :, 0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for j in range(3):
        plt.subplot(1, 3, j + 1)
        plt.title(title[j])
        # Normalization
        plt.imshow(display_list[j] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.show()


@tf.function
def train_step(input_image, target, epoch):
    noise = tf.random.normal([num_examples_to_generate, noise_dim])
    rdm_input = rdm(noise)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator([input_image, rdm_input], training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            train_step(input_image, target, epoch)
        print()

        # Save the model each 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
    for example_input, example_target in test_ds.take(1):
        noise = tf.random.normal([num_examples_to_generate, noise_dim])
        rdm_input = rdm(noise)
        generate_images(generator, [example_input, rdm_input], example_target)

# Data parameters
BUFFER_SIZE = 95
BATCH_SIZE = 1
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_DEPTH = 1
# Path of the dataset
PATH = "/media/jimmyyang/Ubuntu 18.0/ds/sum128new/"

train_dataset = tf.data.Dataset.list_files(PATH + '*.png')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH + '*.png').take(5)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 1

num_examples_to_generate = 1
noise_dim = 100

rdm = random_input()

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator = make_generator()
discriminator = make_discriminator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Address to save the model
checkpoint_dir = '/media/jimmyyang/Ubuntu 18.0/Save_Pix2Pix_addnoise_Defect_Flaw0_128new'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

log_dir = "/media/jimmyyang/Ubuntu 18.0/Summary_Pix2Pix_addnoise_Defect_Flaw0_128new/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

EPOCHS = 200

# Load the model for continuing training, if needed
# checkpoint.restore('/media/jimmyyang/Ubuntu 18.0/Save_Pix2Pix_addnoise_Defect_Flaw0_128new/ckpt-14')

# Start training
fit(train_dataset, EPOCHS, test_dataset)
