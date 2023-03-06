import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

img_height = 100
img_width = 100
batch_size = 25

model = keras.Sequential([
    layers.Conv2D(16,(3,3),activation='relu',input_shape=(100,100,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(16,(3,3),activation='relu'),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(3,activation='softmax')
])

ds_train = keras.preprocessing.image_dataset_from_directory(
    'data/',
    labels='inferred',
    label_mode = 'int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=True,
    seed=100,
    validation_split=0.1,
    subset='training',
)


ds_val = keras.preprocessing.image_dataset_from_directory(
    'data/',
    labels='inferred',
    label_mode = 'int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=True,
    seed=100,
    validation_split=0.1,
    subset='validation',
)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=['accuracy']
)

model.fit(ds_train,epochs=10)