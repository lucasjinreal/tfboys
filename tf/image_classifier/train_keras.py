"""
A simple image classifier on Keras
"""
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os


def build_model_classifier(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def train():
    # params
    target_image_shape = (150, 150)
    batch_size = 16
    num_classes = 5
    img_dir = '../data/flower_photos'

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        img_dir,  # Target Dir
        target_size=target_image_shape,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        img_dir,
        target_size=target_image_shape,
        batch_size=batch_size,
        class_mode='categorical')

    model = build_model_classifier((target_image_shape[0], target_image_shape[1], 3), num_classes)
    try:

        model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=100,
            validation_data=validation_generator,
            validation_steps=800 // batch_size)
        model.save_weights('keras_model_flower.h5')
    except KeyboardInterrupt:
        model.save_weights('keras_model_flower.h5')


if __name__ == '__main__':
    train()