#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import tensorflow as tf
from tensorflow import keras


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [128, 128])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def getImagePaths(paths):
    paths = list(paths.glob('*/*'))
    paths = [str(path) for path in paths]
    random.shuffle(paths)
    return paths


def getLabels(root, paths):
    label_names = sorted(
        item.name for item in root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index)
                          for index, name in enumerate(label_names))
    return [label_to_index[pathlib.Path(path).parent.name]
            for path in paths]


def prepareImages(paths):
    images = list()
    for path in paths:
        pic = load_and_preprocess_image(path)
        picArray = np.array(pic)
        result = list()
        for pixelArray in picArray:
            result2 = list()
            for pixel in pixelArray:
                result2.append(pixel[0])
            result.append(result2)
        images.append(np.array(result))
    return np.array(images)


# Data sources
data_train_root = pathlib.Path("./data/train")
data_validation_root = pathlib.Path("./data/validation")

# Image paths
train_image_paths = getImagePaths(data_train_root)
validation_image_paths = getImagePaths(data_validation_root)

# Image labels
train_image_labels = getLabels(data_train_root, train_image_paths)
validation_image_labels = getLabels(
    data_validation_root, validation_image_paths)

# Images
train_images = prepareImages(train_image_paths)
validation_images = prepareImages(validation_image_paths)

# Inspect preprocess of image
"""plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
"""

# Verify data set
"""
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(label_names[all_image_labels[i]])
plt.show()"""

# Set up the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128, 128)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_image_labels, epochs=10)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(
    validation_images, validation_image_labels)
print('\nTest accuracy:', test_acc)

# Predict
class_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
predictions = model.predict(validation_images)
print("Prediction: ", class_names[np.argmax(predictions[0])])
print("Actual: ",
      class_names[validation_image_labels[np.argmax(predictions[0])]])
