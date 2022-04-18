import tensorflow as tf
import pathlib
import MakeDataset

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Make the GPU(MX150) compatible
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def make_cnn():
    initializer = tf.random_normal_initializer(0., 0.02)
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer=initializer, use_bias=False,
                            input_shape=(128, 128, 1)))
    model.add(layers.MaxPooling2D((2, 2)))  # (?, 64, 64, 8)
    model.add(layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False,
                            activation='relu'))  # (?, 32, 32, 16)
    model.add(layers.MaxPooling2D((2, 2)))  # (?, 16, 16, 16)
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False,
                            activation='relu'))  # (?, 8, 8, 32)
    model.add(layers.MaxPooling2D((2, 2)))  # (?, 4, 4, 32)
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False,
                            activation='relu'))  # (?, 2, 2, 64)

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))

    return model


cnn = make_cnn()
cnn.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])


data_root_test = pathlib.Path('/media/jimmyyang/Ubuntu 18.0/dis_test128new/ds_test')
data_root_real = pathlib.Path('/media/jimmyyang/Ubuntu 18.0/dis_test128new/ds_train_realonly')
data_root_mix_1 = pathlib.Path('/media/jimmyyang/Ubuntu 18.0/dis_test128new/ds_train_mix_1_gen800epc')
data_root_mix_2 = pathlib.Path('/media/jimmyyang/Ubuntu 18.0/dis_test128new/ds_train_mix_2_gen800epc')
data_root_mix_3 = pathlib.Path('/media/jimmyyang/Ubuntu 18.0/dis_test128new/ds_train_mix_3_gen400epc')
data_root_mix_4 = pathlib.Path('/media/jimmyyang/Ubuntu 18.0/dis_test128new/ds_train_mix_4_gen400epc')
data_root_mix_5 = pathlib.Path('/media/jimmyyang/Ubuntu 18.0/dis_test128new/ds_train_mix_5_gen400epc')
data_root_mix_6 = pathlib.Path('/media/jimmyyang/Ubuntu 18.0/dis_test128new/ds_train_mix_6_gen400epc')

image_label_test = MakeDataset.MakeSet(data_root_test)

# Choose a dataset to train the CNN from "data_root_real","data_root_mix_1"...,"data_root_mix_6"
image_label_train = MakeDataset.MakeSet(data_root_real)

train_dataset = image_label_train.shuffle(800).batch(800)
test_dataset = image_label_test.shuffle(200).batch(200)
for batch_test in test_dataset:
    test_image, test_label = batch_test
    break
for batch_train in train_dataset:
    train_image, train_label = batch_train

history = cnn.fit(train_image, train_label, batch_size=10, epochs=200,
                  validation_data=(test_image, test_label))
# plt.plot()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = cnn.evaluate(test_image, test_label, verbose=2)

print(test_acc)
