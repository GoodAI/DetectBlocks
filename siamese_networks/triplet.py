import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow.keras import Model

import argparse
import glob
import image_augmentation_functions as aug


parser = argparse.ArgumentParser()
parser.add_argument("opengl_data_path")
parser.add_argument("blenderproc_data_path")
parser.add_argument("--batch_size", type=int, default=60)
parser.add_argument("--checkpoint_path", type=str, default="/tmp")
parser.add_argument("--chkpt_epoch", type=int, default=-1)
parser.add_argument("--epochs", type=int, default=1000)
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
    if arguments.augment:
        image = aug.add_background(image, bg)
    return image

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

def augment_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    # image = tf.expand_dims(image, axis=0)
    # image = aug.zoom_image_object(
    #     image, np.linspace(0.8, 1.0, 50).astype(np.float32))
    image = aug.add_background(image, next(bg_img_init))
    # image = aug.random_brightness(image, 0.2)
    # image = aug.multiply_brightness(image, [0.6, 1.4])
    # image = aug.contrast_normalization(image, [0.5, 2.0])
    return image

def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """
    bg1 = next(bg_img_init)
    # bg2 = next(bg_img_init)
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


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

# # if arguments.augment:
# a_dataset = a_dataset.map(augment_image)
# # else:
# #     a_dataset = a_dataset.map(preprocess_image)
# p_dataset = p_dataset.map(augment_image)
# n_dataset = n_dataset.map(augment_image)

dataset = tf.data.Dataset.zip((a_dataset, p_dataset, n_dataset))
dataset = dataset.map(preprocess_triplets)
# dataset = dataset.shuffle(buffer_size=image_count, reshuffle_each_iteration=False)
# dataset = dataset.cache("{}/checkpoints/triplet/{}/cache_full".format(root, nm))
dataset = dataset.shuffle(buffer_size=1000)

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.9))
val_dataset = dataset.skip(round(image_count * 0.9))

train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)


def visualize(anchor, positive, negative, name):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(12, 6))
    axs = fig.subplots(3, 6)

    for j in range(1):
        for i in range(6):
            if 16*j+i < batch_size:
                show(axs[3*j, i], anchor[16*j+i])
                show(axs[3*j+1, i], positive[16*j+i])
                show(axs[3*j+2, i], negative[16*j+i])

    plt.savefig('/tmp/triplet_{}_batch.png'.format(nm), bbox_inches='tight')
    print('Saved example batch to /tmp/triplet_{}_batch.png'.format(nm))


if arguments.vis:
    visualize(*list(train_dataset.take(1).as_numpy_iterator())[0], nm)


if arguments.simple_network:
    input = layers.Input((128, 128, 3))
    x = layers.BatchNormalization()(input)
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
    embedding = Model(input, x, name="embed_network")
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
    x = layers.Dense(512, activation="softplus")(x)
    embedding = Model(input, x, name="embed_network")


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        # ap_distance = 1 + losses.CosineSimilarity()(anchor, positive)
        # an_distance = 1 + losses.CosineSimilarity()(anchor, negative)
        # ap_distance = tf.sigmoid(tf.reduce_sum(tf.square(anchor - positive), -1))
        # an_distance = tf.sigmoid(tf.reduce_sum(-tf.square(anchor - negative), -1))
        a_norm = tf.norm(anchor, axis=1)
        p_norm = tf.norm(positive, axis=1)
        n_norm = tf.norm(negative, axis=1)
        ap_dot = tf.matmul(anchor, tf.transpose(positive))
        an_dot = tf.matmul(anchor, tf.transpose(negative))
        ap_distances = \
            tf.expand_dims(a_norm, 0) - 2 * ap_dot + tf.expand_dims(p_norm, 1)
        an_distances = \
            tf.expand_dims(a_norm, 0) - 2 * an_dot + tf.expand_dims(n_norm, 1)
        ap_max_dists = tf.reduce_max(ap_distances, axis=1, keepdims=True)
        an_min_dists = tf.reduce_min(an_distances, axis=1, keepdims=True)
        return (ap_max_dists, an_min_dists)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

if arguments.simple_network:
    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )
else:
    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.p_loss_tracker = metrics.Mean(name="p_loss")
        self.n_loss_tracker = metrics.Mean(name="n_loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss, p_loss, n_loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        self.p_loss_tracker.update_state(p_loss)
        self.n_loss_tracker.update_state(n_loss)
        return {"loss": self.loss_tracker.result(),
                "p_loss": self.p_loss_tracker.result(),
                "n_loss": self.n_loss_tracker.result()}

    def test_step(self, data):
        loss, p_loss, n_loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        self.p_loss_tracker.update_state(p_loss)
        self.n_loss_tracker.update_state(n_loss)
        return {"loss": self.loss_tracker.result(),
                "p_loss": self.p_loss_tracker.result(),
                "n_loss": self.n_loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_max_dists, an_min_dists = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = tf.reduce_mean(
            tf.maximum(ap_max_dists - an_min_dists + self.margin, 0.0))
        return loss, tf.reduce_mean(ap_max_dists), tf.reduce_mean(an_min_dists)

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


siamese = SiameseModel(siamese_network, margin=1000.0)
siamese.compile(optimizer=optimizers.Adam(arguments.lr))

checkpoint_dir = os.path.join(root, "checkpoints/triplet", nm)
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
        epochs=epochs,
        callbacks=[cp_callback],
    )

if arguments.save_model:
    embedding.compile(optimizer=optimizers.Adam(arguments.lr))
    save_path = os.path.join(checkpoint_dir,
        '{}_{}.h5'.format(nm, arguments.chkpt_epoch))
    embedding.save(save_path)
    print("Model saved to {}".format(save_path))
