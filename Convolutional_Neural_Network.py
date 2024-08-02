# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 01:28:48 2024

@author: Ayshika Kapoor
"""

#importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
# Adding a convolution layer with 16 filters, each of size 3x3
# Input shape is set to (128, 128, 3) for 128x128 RGB images
classifier.add(Convolution2D(16, (3, 3), input_shape=(128, 128, 3), activation='relu'))

# Step 2 - Pooling
# Adding a max pooling layer with a pool size of 2x2
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding another convolution layer
classifier.add(Convolution2D(16, (3, 3), activation='relu'))
# Adding another max pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a third convolution layer
classifier.add(Convolution2D(16, (3, 3), activation='relu'))
# Adding a third max pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
# Flattening the pooled feature maps into a single vector
classifier.add(Flatten())

# Step 4 - Full Connection
# Adding a fully connected layer with 64 units and ReLU activation
classifier.add(Dense(units=64, activation='relu'))
# Adding the output layer with a single unit and sigmoid activation for binary classification
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
# Using Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Preparing the image data generators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Rescale pixel values to [0, 1]
    shear_range=0.2,          # Apply shear transformations
    zoom_range=0.2,           # Apply zoom transformations
    horizontal_flip=True      # Allow horizontal flipping
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]

# Creating the training data generator
training_set = train_datagen.flow_from_directory(
    '/Train',                 # Directory of the training set
    target_size=(128, 128),   # Resize images to 128x128
    batch_size=32,            # Number of images per batch
    class_mode='binary'       # Binary classification
)

# Creating the testing data generator
testing_set = test_datagen.flow_from_directory(
    '/Test',                  # Directory of the testing set
    target_size=(128, 128),   # Resize images to 128x128
    batch_size=32,            # Number of images per batch
    class_mode='binary'       # Binary classification
)

# Training the CNN
# Using the training set, with 1 step per epoch and 50 epochs
# Using the testing set for validation with 40 validation steps
classifier.fit_generator(
    training_set,
    steps_per_epoch=1,
    epochs=50,
    validation_data=testing_set,
    validation_steps=40
)
