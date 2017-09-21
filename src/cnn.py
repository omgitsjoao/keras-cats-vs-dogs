# Convolutional Network for Cat Detection

from time import time

from augmentation import dev_augment, train_augment
from nets import shallow_net

BATCH_SIZE = 32

t_augment = train_augment()
d_augment = dev_augment()
model = shallow_net()

train_generator = t_augment.flow_from_directory('./data/train',
                                                target_size=(150, 150),
                                                batch_size=BATCH_SIZE,
                                                class_mode='binary')

dev_generator = d_augment.flow_from_directory('./data/dev',
                                              target_size=(150, 150),
                                              batch_size=BATCH_SIZE,
                                              class_mode='binary')

model.fit_generator(train_generator,
                    steps_per_epoch=250,
                    epochs=50,
                    validation_data=dev_generator,
                    validation_steps=150)

model.save_weights('./weights/{}_weights.h5'.format(time()))
