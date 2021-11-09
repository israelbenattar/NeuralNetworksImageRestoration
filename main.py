import re
import os, itertools, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
from skimage.draw import line

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

from imageio import imread
from skimage.color import rgb2gray
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, UpSampling2D, Dense, Flatten, Reshape, \
    AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage import color


def relpath(path):
    """Returns the relative path to the script's location

    Arguments:
    path -- a string representation of a path.
    """
    return os.path.join(os.getcwd(), path)


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.

    Arguments:
    path -- path to a directory to search for images.
    use_shuffle -- option to shuffle order of files. Uses a fixed shuffled order.
    """

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']

    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """Returns a list of image paths to be used for image denoising in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def images_for_deblurring():
    """Returns a list of image paths to be used for text deblurring in Ex5"""
    return list_images(relpath("current/text_dataset/train"), True)


def images_for_super_resolution():
    """Returns a list of image paths to be used for image super-resolution in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.

    Arguments:
    kernel_size -- the height and width of the kernel. Controls strength of blur.
    angle -- angle in the range [0, np.pi) for the direction of the motion.
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size - 1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2 * half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1 - norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2 * half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel


def read_image(filename, representation):
    """Reads an image, and if needed makes sure it is in [0,1] and in float64.
    arguments:
    filename -- the filename to load the image from.
    representation -- if 1 convert to grayscale. If 2 keep as RGB.
    """
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im).astype(np.float64)
    if im.dtype == np.uint8:
        im = im.astype(np.float64) / 255.0
    return im


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator for generating pairs of image patches, corrupted and original
    :param filenames: a list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy array representation of an image as a single argument, and returning a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return:outputs random tuples of the form (source_batch, target_batch), where each output variable is an array of shape(batch_size, height, width, 1).
     target_batch is made of clean images and source_batch is their respective randomly corrupted version
     according to corruption_func(im)
    """
    cached_images = {}
    while True:
        source_batch = np.empty((batch_size, crop_size[0], crop_size[1], 1))
        target_batch = np.empty((batch_size, crop_size[0], crop_size[1], 1))
        rand_image = np.random.choice(filenames, batch_size)
        for index, image_name in enumerate(rand_image):
            if image_name in cached_images:
                image = cached_images[image_name]
            else:
                image = read_image(image_name, 1)
                cached_images[image_name] = image
            max_x, max_y = crop_size[0] * 3, crop_size[1] * 3
            x = np.random.randint(0, image.shape[0] - max_x)
            y = np.random.randint(0, image.shape[1] - max_y)
            patch = image[x:x + max_x, y:y + max_y]
            blurred_patch = corruption_func(patch)
            x = np.random.randint(0, patch.shape[0] - crop_size[0])
            y = np.random.randint(0, patch.shape[1] - crop_size[1])
            source_batch[index, :, :, 0] = patch[x:x + crop_size[0], y:y + crop_size[1]] - .5
            target_batch[index, :, :, 0] = blurred_patch[x:x + crop_size[0], y:y + crop_size[1]] - .5

        yield target_batch, source_batch


def resblock(input_tensor, num_channels):
    """
    Takes as input a symbolic input tensor and the number of channels for each of its convolutional layers, and returns the symbolic output tensor of the resnet block.
    The convolutional layers should use “same” border mode, so as to not decrease the spatial dimension of the output tensor.
    :param input_tensor: input tensor
    :param num_channels: number of channels
    :return: symbolic output tensor of the resnet block
    """
    x = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(num_channels, (3, 3), padding='same')(x)
    x = Add()([input_tensor, x])
    x = Activation('relu')(x)
    return x


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Create an untrained Keras model with input dimension the shape of (height, width, 1), and all convolutional layers (including residual
    blocks) with number of output channels equal to num_channels, except the very last convolutional layer which should have a single output channel.
    The number of residual blocks should be equal to num_res_blocks.
    :param height: height
    :param width: width
    :param num_channels: number of channels
    :param num_res_blocks: number of residual blocks
    :return: an untrained Keras model.
    """
    a = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, (3, 3), padding='same')(a)
    b = Activation('relu')(b)
    for _ in range(num_res_blocks):
        b = resblock(b, num_channels)
    b = Conv2D(1, (3, 3), padding='same')(b)
    b = Add()([a, b])
    return Model(inputs=a, outputs=b)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Divide the images into a training set and validation set, using an 80-20 split, and generate from each set a dataset with the given batch size
    and corruption function. Eventually it will train the model.
    :param model:  a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and should append anything to them.
    :param corruption_func: a corruption function.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: the number of update steps in each epoch.
    :param num_epochs: the number of epochs for which the optimization will run.
    :param num_valid_samples: the number of samples in the validation set to test on after every epoch.
    """
    training_dataset, validation_dataset = np.split(images, [int(len(images) * .8)])
    crop_size = (model.input_shape[1], model.input_shape[2])
    training_generator = load_dataset(training_dataset, batch_size, corruption_func, crop_size)
    validation_generator = load_dataset(validation_dataset, batch_size, corruption_func, crop_size)
    optimizer = Adam(beta_2=0.9)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=validation_generator, validation_steps=num_valid_samples)


"""# 6 Image Restoration of Complete Images"""


def restore_image(corrupted_image, base_model):
    """
    Restore full images of any size
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the [0, 1] range of type float64 that is affected
    by a corruption generated from the same corruption function encountered during training (the image is not necessarily from the training set though).
    :param base_model: a neural network trained to restore small patches. The input and output of the network are images with values in the [−0.5, 0.5] range.
    :return: the restored image
    """
    s = corrupted_image.shape
    corrupted_image = corrupted_image.reshape((corrupted_image.shape[0], corrupted_image.shape[1], 1))
    a = Input(shape=corrupted_image.shape)
    new_model = Model(inputs=a, outputs=base_model(a))
    restored_image = new_model.predict((corrupted_image[np.newaxis, ...] - .5))[0].astype('float64')
    restored_image = restored_image.reshape(s)
    return np.clip(restored_image + .5, 0, 1)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Add random gaussian noise to an image
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal variance of the gaussian distribution
    :return: the corrupted image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    norm = np.random.normal(0, sigma, image.shape)
    noise_image = np.round((image + norm) * 255) / 255
    return np.clip(noise_image, 0, 1)


def learn_denoising_model(denoise_num_res_blocks, quick_mode=False):
    """
    Train a denoising model
    :param denoise_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    batch_size, steps_per_epoch, num_epochs, num_valid_samples = 100, 100, 10, 1000
    if quick_mode: batch_size, steps_per_epoch, num_epochs, num_valid_samples = 10, 3, 2, 30
    images = images_for_denoising()
    model = build_nn_model(24, 24, 48, denoise_num_res_blocks)
    train_model(model, images, lambda x: add_gaussian_noise(x, 0, 0.2), batch_size, steps_per_epoch, num_epochs,
                num_valid_samples)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size:  an odd integer specifying the size of the kernel.
    :param angle: an angle in radians in the range [0, π).
    :return: blurred image
    """
    kernel = motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: blurred image
    """
    angle = np.random.uniform(low=0, high=np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes, 1)[0]
    return add_motion_blur(image, kernel_size, angle)


def learn_deblurring_model(deblur_num_res_blocks, quick_mode=False):
    """
    Train a deblurring model
    :param deblur_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    batch_size, steps_per_epoch, num_epochs, num_valid_samples = 100, 100, 10, 1000
    if quick_mode: batch_size, steps_per_epoch, num_epochs, num_valid_samples = 10, 3, 2, 30
    images = images_for_deblurring()
    model = build_nn_model(16, 16, 32, deblur_num_res_blocks)
    train_model(model, images, lambda x: random_motion_blur(x, [7]), batch_size, steps_per_epoch, num_epochs,
                num_valid_samples)
    return model


def super_resolution_corruption(image):
    """
    Perform the super resolution corruption
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :return: corrupted image
    """
    factor = np.random.randint(2, 4)
    width, height = image.shape
    out_image = np.copy(image)
    image_1 = out_image[:(width // factor) * factor, :(height // factor) * factor]
    image_1 = zoom(image_1, 1 / factor, mode='wrap')
    image_1 = zoom(image_1, factor, mode='wrap')
    out_image[:(width // factor) * factor, :(height // factor) * factor] = image_1
    return out_image.clip(0, 1)


def learn_super_resolution_model(super_resolution_num_res_blocks, quick_mode=False):
    """
    Train a super resolution model
    :param super_resolution_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    batch_siz, steps_per_epoc, num_epoch, num_valid_samples = batch_size, steps_per_epoch, num_epochs, 1000
    if quick_mode: batch_siz, steps_per_epoc, num_epoch, num_valid_samples = 10, 3, 2, 30
    images = images_for_super_resolution()
    model = build_nn_model(patch_size, patch_size, num_channels, super_resolution_num_res_blocks)
    train_model(model, images, lambda x: super_resolution_corruption(x), batch_siz, steps_per_epoc, num_epoch,
                num_valid_samples)
    return model
