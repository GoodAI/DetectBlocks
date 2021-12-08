import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import argparse
import os
import glob
import image_augmentation_functions as aug
from pathlib import Path
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.applications import resnet


parser = argparse.ArgumentParser()
parser.add_argument("opengl_data_path")
parser.add_argument("blenderproc_data_path")
parser.add_argument("--batch_size", type=int, default=60)
parser.add_argument("--checkpoint_path", type=str, default="/tmp")
parser.add_argument("--chkpt_epoch", type=int, default=-1)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--imgs_per_model", type=int, default=7998)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--num_models", type=int, default=16)
parser.add_argument("-eval", action="store_true", default=False)
parser.add_argument("-save_model", action="store_true", default=False)
parser.add_argument("-simple_network", action="store_true", default=False)
parser.add_argument("-train_from_chkpt", action="store_true", default=False)
parser.add_argument("-vis", action="store_true", default=False)
arguments = parser.parse_args()
nm = arguments.exp_name

batch_size = arguments.batch_size
epochs = arguments.epochs
root = arguments.checkpoint_path
target_shape = (128, 128)

margin = 1


def load_bg_imgs(in_path):
    return tf.image.resize(tf.image.convert_image_dtype(
        tf.image.decode_jpeg(tf.io.read_file(in_path)), tf.float32),
        [128, 128])

def load_random_color(in_path):
    return tf.concat([
        tf.fill([128, 128, 1], tf.random.uniform([], dtype=tf.float32)),
        tf.fill([128, 128, 1], tf.random.uniform([], dtype=tf.float32)),
        tf.fill([128, 128, 1], tf.random.uniform([], dtype=tf.float32))
    ], axis=2)

def load_purple(in_path):
    return tf.convert_to_tensor(np.full((128, 128, 3),
        [200/256, 0, 200/256]), dtype=tf.float32)


bg_paths = glob.glob("/volume/pekdat/datasets/public/VOCdevkit/original/VOC2012/JPEGImages/*.jpg")
bg_imgs_dataset = tf.data.Dataset.from_tensor_slices(bg_paths)
bg_imgs_dataset = bg_imgs_dataset.map(map_func=load_bg_imgs)
bg_imgs_dataset = bg_imgs_dataset.shuffle(1000)
bg_imgs_dataset = bg_imgs_dataset.repeat()
bg_imgs_dataset = bg_imgs_dataset.prefetch(1)

bg_img_init = iter(bg_imgs_dataset)


def add_background(image, bg):
    image = aug.add_background(image, bg)
    return image

def augment_image(image):
    image = tf.expand_dims(image, axis=0)
    image = aug.zoom_image_object(
        image, np.linspace(0.5, 1.0, 50).astype(np.float32))
    image = aug.add_background(image, next(bg_img_init))
    image = aug.random_brightness(image, 0.2)
    image = aug.multiply_brightness(image, [0.6, 1.4])
    image = aug.contrast_normalization(image, [0.5, 2.0])
    return image[0]

def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

def augment_pair(image_tuple, label):
    anchor, image = image_tuple
    bg1 = next(bg_img_init)
    # bg2 = next(bg_img_init)
    if label == 1.0:
        return ((
            preprocess_image(anchor),
            preprocess_image(image),
        ), label)
    return ((
        preprocess_image(anchor),
        preprocess_image(image),
    ), label)
        

# We need to make sure both the anchor and positive images are loaded in
# sorted order so we can match them together.
anchor_images = sorted(
    [os.path.join(root, file) for root, dirs, files in \
        os.walk(os.path.join(arguments.opengl_data_path, 'train_y')) \
        for file in files]
)
# positive_images = sorted(
#     [os.path.join(root, file) for root, dirs, files in \
#         os.walk(arguments.blenderproc_data_path) for file in files]
# )
positive_images = anchor_images

image_count = len(anchor_images)
imgs_per_model = arguments.imgs_per_model
num_models = arguments.num_models

shuffled_images = positive_images
shuffled_images = np.asarray(shuffled_images)
for i in range(num_models):
    np.random.RandomState().shuffle(shuffled_images[imgs_per_model*i:imgs_per_model*(i+1)])

# for each stage we take a third of images from each of the 3 stages left and
# form a batch of negative examples (where block matches but stage does not).
# the negative examples must be in the same pose
imgs_per_block = imgs_per_model * 4
stages_per_block = 4
num_blocks = num_models // stages_per_block
third_of_model = imgs_per_model // 3

negative_images = anchor_images
negative_images = np.array(negative_images)
for i in range(num_blocks):
    block_start = imgs_per_block * i
    for j in range(stages_per_block):
        stage_start = block_start + imgs_per_model * j
        for k in range(3):
            third_start = stage_start + third_of_model * k
            third_end = third_start + third_of_model
            anchor_third_start = block_start + (imgs_per_model * j +
                third_of_model * k + imgs_per_model * (k+1)) % imgs_per_block
            anchor_third_end = anchor_third_start + third_of_model
            negative_images[third_start:third_end] = \
                anchor_images[anchor_third_start:anchor_third_end]

np.random.RandomState(seed=42).shuffle(anchor_images)
np.random.RandomState(seed=42).shuffle(shuffled_images)
np.random.RandomState(seed=42).shuffle(negative_images)

a_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
p_dataset = tf.data.Dataset.from_tensor_slices(shuffled_images)
n_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

true_labels = tf.data.Dataset.from_tensor_slices([[1.0] for i in range(image_count)])
false_labels = tf.data.Dataset.from_tensor_slices([[0.0] for i in range(image_count)])

ap_dataset = tf.data.Dataset.zip((tf.data.Dataset.zip((a_dataset, p_dataset)), true_labels))
an_dataset = tf.data.Dataset.zip((tf.data.Dataset.zip((a_dataset, n_dataset)), false_labels))

dataset = tf.data.experimental.sample_from_datasets([ap_dataset, an_dataset], weights=[0.5, 0.5])
dataset = dataset.map(augment_pair)
# dataset = dataset.cache("{}/checkpoints/contrastive/{}/cache_full".format(root, nm))
dataset = dataset.shuffle(buffer_size=1000)

train_dataset = dataset.take(round(2 * image_count * 0.8))
val_dataset = dataset.skip(round(2 * image_count * 0.8))

train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)


def visualize_dataset(image_tuple, label, name):
    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    anchor, image = image_tuple

    fig = plt.figure(figsize=(12, 6))
    axs = fig.subplots(3, 6)

    for j in range(1):
        for i in range(6):
            if 15*j+i < batch_size:
                show(axs[3*j, i], anchor[15*j+i])
                show(axs[3*j+1, i], image[15*j+i])
                color = np.array([0, 255, 0], dtype=int) \
                    if label[15*j+i] == 1.0 \
                    else np.array([255, 0, 0], dtype=int)
                label_img = np.full_like(anchor[15*j+i], color)
                show(axs[3*j+2, i], label_img)

    plt.savefig('/tmp/contr_{}_batch.png'.format(nm), bbox_inches='tight')
    print('Saved example batch to /tmp/contr_{}_batch.png'.format(nm))


if arguments.vis:
    visualize_dataset(*list(train_dataset.take(1).as_numpy_iterator())[0], nm)


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


if arguments.simple_network:
    input1 = layers.Input((128, 128, 3))
    input2 = layers.Input((128, 128, 3))

    x = tf.keras.layers.BatchNormalization()(input1)
    x = layers.Conv2D(4, (5, 5), activation="tanh",
        kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(16, (5, 5), activation="tanh",
        kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="tanh",
        kernel_regularizer=regularizers.l2(0.01))(x)
    embedding_network1 = keras.Model(input1, x, name="embed_network")

    y = tf.keras.layers.BatchNormalization()(input2)
    y = layers.Conv2D(4, (5, 5), activation="tanh",
        kernel_regularizer=regularizers.l2(0.001))(y)
    y = layers.AveragePooling2D(pool_size=(2, 2))(y)
    y = layers.Conv2D(16, (5, 5), activation="tanh",
        kernel_regularizer=regularizers.l2(0.001))(y)
    y = layers.AveragePooling2D(pool_size=(2, 2))(y)
    y = layers.Flatten()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = layers.Dense(10, activation="tanh",
        kernel_regularizer=regularizers.l2(0.001))(y)
    embedding_network2 = keras.Model(input2, y)
else:
    input = layers.Input((128, 128, 3))
    x = input
    num_filters = [128, 256, 512, 512]
    strides = [2, 2, 2, 2]
    for filters, stride in zip(num_filters, strides):
        padding = "same"
        x = layers.Conv2D(
            filters=filters,
            kernel_size=5,
            strides=stride,
            padding=padding,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            activation="relu"
        )(x)
        x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="softplus")(x)
    embedding_network1 = keras.Model(input, x, name="embed_network")

input_1 = layers.Input((128, 128, 3))
input_2 = layers.Input((128, 128, 3))

tower_1 = embedding_network1(input_1)
tower_2 = embedding_network1(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)


def loss(margin=1):
    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


siamese.compile(loss=loss(margin=margin),
                optimizer=optimizers.Adam(arguments.lr), metrics=["accuracy"])
siamese.summary()


checkpoint_dir = os.path.join(root, "checkpoints/contrastive", nm)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, "{epoch:04d}.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

if arguments.eval or arguments.train_from_chkpt:
    siamese.load_weights(checkpoint_path.format(epoch=arguments.chkpt_epoch))
if not arguments.eval:
    history = siamese.fit(
        train_dataset,
        validation_data=val_dataset,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[cp_callback],
    )

if arguments.save_model:
    embedding_network1.compile(optimizer=optimizers.Adam(arguments.lr))
    save_path = os.path.join(checkpoint_dir,
        '{}_{}.h5'.format(nm, arguments.chkpt_epoch))
    embedding_network1.save(save_path)
    print("Model saved to {}".format(save_path))
