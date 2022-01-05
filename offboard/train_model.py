"""
preprocess images nad train lane navigation model
"""
import fnmatch
import os
import pickle
import random
from os import path
from os.path import exists
from os.path import join

import numpy as np

np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})

import pandas as pd
pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('display.max_colwidth', 200)

# tensorflow
import tensorflow
import tensorflow.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# sklearn
from sklearn.model_selection import train_test_split

# imaging
import matplotlib.pyplot as plt
from PIL import Image

from util import constants
from util.image_processing import pan
from util.image_processing import zoom
from util.image_processing import blur
from util.image_processing import adjust_brightness
from util.image_processing import my_imread
from util.image_processing import preprocess_image


def get_angle_from_filename(filename):
    return int(filename[-11:-8])


def load_data(data_directory):
    dir_list = os.listdir(data_directory)
    image_paths = []
    steering_angles = []
    pattern = "*.png"
    for d in dir_list:
        if path.isdir(join(data_directory, d)):
            filenames = os.listdir(join(data_directory, d))
            for filename in filenames:
                if fnmatch.fnmatch(filename, pattern):
                    image_paths.append(os.path.join(data_directory, d, filename))
                    steering_angles.append(get_angle_from_filename(filename))
    return image_paths, steering_angles


def nvidia_model():
    model = Sequential(name="Nvidia_Model")
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    # model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=constants.LEARNING_RATE)
    # model = tfmot.quantization.keras.quantize_model(model) todo: figure out quantization
    model.compile(loss='mse', optimizer=optimizer)
    return model


def random_augment(image, steering_angle):
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = blur(image)
    if np.random.rand() < 0.5:
        image = adjust_brightness(image)
    # image, steering_angle = random_flip(image, steering_angle)  # may be flip is not helping

    return image, steering_angle


def image_data_generator(image_paths, steering_angles, batch_size, is_training):
    while True:
        batch_images = []
        batch_steering_angles = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            image = my_imread(image_paths[random_index])
            steering_angle = steering_angles[random_index]
            if is_training:
                # training: augment image
                image, steering_angle = random_augment(image, steering_angle)

            image = preprocess_image(image)
            batch_images.append(image)
            batch_steering_angles.append(steering_angle)

        yield np.asarray(batch_images, dtype=np.float32), np.asarray(batch_steering_angles)


def test_data(image_paths, steering_angles, x_train, x_valid, y_train, y_valid):
    image_index = random.randint(0, len(image_paths) - 1)
    plt.imshow(Image.open(image_paths[image_index]))
    print("image_path: %s" % image_paths[image_index])
    print("steering_Angle: %d" % steering_angles[image_index])

    df = pd.DataFrame()
    df['ImagePath'] = image_paths
    df['Angle'] = steering_angles
    num_of_bins = 25

    # hist, bins = np.histogram(df['Angle'], num_of_bins)
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    axes.hist(df['Angle'], bins=num_of_bins, width=1, color='blue')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_train, bins=num_of_bins, width=1, color='blue')
    axes[0].set_title('Training Data')
    axes[1].hist(y_valid, bins=num_of_bins, width=1, color='red')
    axes[1].set_title('Validation Data')

    while (True):
        fig, axes = plt.subplots(1, 6, figsize=(15, 10))
        image_orig = my_imread(image_paths[image_index])
        image_zoom = zoom(image_orig)
        image_pan = pan(image_orig)
        image_brightness = adjust_brightness(image_orig)
        image_blur = blur(image_orig)
        image_processed = preprocess_image(image_orig)
        axes[0].imshow(image_orig)
        axes[0].set_title("orig")

        axes[1].imshow(image_zoom)
        axes[1].set_title("zoomed")

        axes[2].imshow(image_pan)
        axes[2].set_title("Panned")

        axes[3].imshow(image_brightness)
        axes[3].set_title("Brightness")

        axes[4].imshow(image_blur)
        axes[4].set_title("Blurred")

        axes[5].imshow(image_processed)
        axes[5].set_title("processed")
        plt.show()
        key = input("Press a key to continue")
        if (key == 'q'):
            quit()
        elif (key == 'c'):
            break
        image_index = random.randint(0, len(image_paths) - 1)


def main():
    data_dir = os.path.join(constants.ROOT_FOLDER, constants.DATA_FOLDER)
    model_dir = os.path.join(constants.ROOT_FOLDER, constants.MODEL_FOLDER)
    final_model_path = join(model_dir, constants.MODEL_FINAL_NAME)
    checkpoint_model_path = join(model_dir, constants.MODEL_CHECKPOINT_NAME)
    image_paths, steering_angles = load_data(data_dir)

    x_train, x_valid, y_train, y_valid = train_test_split(image_paths, steering_angles, test_size=0.2)
    print("# training: %d" % len(x_train))
    print("# valid: %d" % len(x_valid))

    test_data(image_paths, steering_angles, x_train, x_valid, y_train, y_valid)  # press c to continue; q to quit

    # train existing model if exists
    if exists(final_model_path):
        model = load_model(final_model_path)
    elif exists(checkpoint_model_path):
        model = load_model(checkpoint_model_path)
    else:
        model = nvidia_model()
    print(model.summary())

    log_dir_root = f'{model_dir}/logs/'
    checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_model_path, verbose=1, save_best_only=True)
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir_root, histogram_freq=1)

    images, _ = next(image_data_generator(x_train, y_train, batch_size=100, is_training=True))

    # Creates a file writer for the log directory.
    file_writer = tensorflow.summary.create_file_writer(log_dir_root)

    # Using the file writer, log the reshaped image.
    with file_writer.as_default():
        tensorflow.summary.image("100 training data examples", images, max_outputs=100, step=0)

    # let the training begin!
    history = model.fit(image_data_generator(x_train, y_train, batch_size=constants.BATCH_SIZE, is_training=True),
                        steps_per_epoch=constants.STEPS_PER_EPOCH,
                        epochs=constants.NUM_OF_EPOCHS,
                        validation_data=image_data_generator(x_valid, y_valid, batch_size=constants.BATCH_SIZE,
                                                             is_training=False),
                        validation_steps=constants.VALIDATION_STEPS,
                        verbose=1,
                        shuffle=1,
                        callbacks=[checkpoint_callback, tensorboard_callback])

    model.save(final_model_path)

    history_path = os.path.join(model_dir, 'history.pickle')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
