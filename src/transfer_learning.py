# todo create a notebook explaining transfer

import uuid

import numpy as np
from keras.applications import VGG16
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from util import count_files_on

IMG_SIZE = (150, 150)
BATCH_SIZE = 16
COUNT_TRAIN_CATS = count_files_on('./data/train/cats')
COUNT_TRAIN_DOGS = count_files_on('./data/train/dogs')
COUNT_DEV_CATS = count_files_on('./data/dev/cats')
COUNT_DEV_DOGS = count_files_on('./data/dev/dogs')
EPOCH = 50
TRAIN_STEPS = np.sum([COUNT_TRAIN_CATS, COUNT_TRAIN_DOGS]) // BATCH_SIZE
DEV_STEPS = np.sum([COUNT_DEV_CATS, COUNT_DEV_DOGS]) // BATCH_SIZE
# steps is a batch number for how many samples per prediction should be computed by vgg
# it should be a round number or it will predict only a portion of the total, screwing the
# model fit
# TIP USE A ROUND NUMBER OF SAMPLES AND A BATCH SIZE THAT DIVIDES IT W/O REST
RUN_ID = str(uuid.uuid4())


def save_vgg_features():
    """"Use vgg to compute advanced features for our model but don't use it to classify them"""
    datagen = ImageDataGenerator(rescale=1. / 255)
    model = VGG16(include_top=False, weights='imagenet')

    generator = _make_flow_from(datagen, './data/train')

    train_features = model.predict_generator(generator, TRAIN_STEPS)
    np.save('./weights/features_train.{}.npy'.format(RUN_ID), train_features)

    generator = _make_flow_from(datagen, './data/dev')

    dev_features = model.predict_generator(generator, DEV_STEPS)
    np.save('./weights/features_dev.{}.npy'.format(RUN_ID), dev_features)


def train_top_layers():
    train_features = np.load('./weights/features_train.{}.npy'.format(RUN_ID))
    dev_features = np.load('./weights/features_dev.{}.npy'.format(RUN_ID))

    train_labels = _compute_train_labels()
    dev_labels = _compute_dev_labels()
    tensorboard = _tensorboard()

    model = Sequential()
    # transform into 1D array
    model.add(Flatten(input_shape=train_features.shape[1:]))
    # Fully connected layer with 256 neurons
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # Output layer with sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_features, train_labels,
              epochs=EPOCH, batch_size=BATCH_SIZE,
              validation_data=(dev_features, dev_labels),
              callbacks=[tensorboard])

    model.save_weights('./weights/top_layers.{}.h5'.format(RUN_ID))


def _compute_train_labels():
    """"
    Here we need to 'create' the labels from our training set, since we set shuffle to False on
    the flow to generate features we can assume that `m` is the number of items per class folder
    so it comes as m_cats + m_dogs, giving m_cats value of 1 and 0 to others.
    Cats comes first because C comes first in alphabet, thus order
    """
    cats = [1] * COUNT_TRAIN_CATS
    dogs = [0] * COUNT_TRAIN_DOGS
    return np.array(cats + dogs)


def _compute_dev_labels():
    """"See _compute_train_labels"""
    cats = [1] * COUNT_DEV_CATS
    dogs = [0] * COUNT_DEV_DOGS
    return np.array(cats + dogs)


def _tensorboard():
    return TensorBoard(log_dir='./logs',
                       histogram_freq=0,
                       batch_size=BATCH_SIZE,
                       write_graph=True,
                       write_grads=True,
                       write_images=True)


def _make_flow_from(datagen, folder):
    # `shuffle`  is a hack important, it will preserve the order of images
    # from the folders inside train,so all cats come first then dogs.
    # To make easier to reason about I make a function to count how many are inside them.
    return datagen.flow_from_directory(folder,
                                       IMG_SIZE,
                                       batch_size=BATCH_SIZE,
                                       class_mode=None,
                                       shuffle=False)
