from keras.preprocessing.image import ImageDataGenerator


def train_augment():
    return ImageDataGenerator(
            rotation_range=40,  # rotate until 40 deg
            width_shift_range=.2,
            height_shift_range=.2,
            rescale=1. / 255,  # images are from 0 to 255, convert it to 0-1 to minimize computation
            shear_range=.2,
            zoom_range=0.2,
            horizontal_flip=True,
            # vertical_flip=True,
            fill_mode='nearest')  # we can create around 710 images with this configuration for each one we have on training


def dev_augment():
    # only rescale it since we will deal with real data
    return ImageDataGenerator(rescale=1. / 255)
