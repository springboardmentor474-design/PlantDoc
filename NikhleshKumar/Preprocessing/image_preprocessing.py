# -*- coding: utf-8 -*-
"""image_preprocessing.ipynb

Original file is located at
    https://colab.research.google.com/drive/1QX7uxt_qIQJqbR3KVK7HZanNKd74StW4
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from google.colab import drive
drive.mount('/content/drive')

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(

    test_path,

    target_size=(224,224),

    batch_size=32,

    class_mode="categorical"

)

train_path = "/content/drive/MyDrive/PlantChatBot/imag/train"
test_path = "/content/drive/MyDrive/PlantChatBot/imag/test"

train_path = "train"
test_path = "test"

train_datagen = ImageDataGenerator(

    rescale=1./255,          # Normalize pixels

    rotation_range=20,       # Data augmentation
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(

    train_path,

    target_size=(224,224),    # Resize images

    batch_size=32,

    class_mode="categorical"

)

import os
print(os.listdir())

import os
print(os.listdir('/content/drive/MyDrive'))

print(os.listdir('/content/drive/MyDrive/PlantChatBot'))

train_path = "/content/drive/MyDrive/PlantChatBot/train"
test_path = "/content/drive/MyDrive/PlantChatBot/test"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

images, labels = next(train_generator)

print(images.shape)
print(labels.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_path = "/content/drive/MyDrive/PlantChatBot/train"
test_path = "/content/drive/MyDrive/PlantChatBot/test"

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(train_generator.num_classes,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)
