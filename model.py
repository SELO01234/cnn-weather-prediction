from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os.path


class ImageModel:

    def __init__(self):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.model = None

    def build(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(100, 100, 3)))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(4, activation='softmax'))

    def summary(self):
        self.model.summary()

    def compile(self):
        self.model.compile(optimizer= Adam(learning_rate= 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, train_data, train_labels, validation_data, validation_labels):
        self.model.fit(train_data, train_labels, validation_data=(validation_data, validation_labels), epochs= 50, verbose= 2)

        if os.path.isfile('cnn_model.h5') is False:
            self.model.save('cnn_model.h5')

    def predict(self, test_data):
        return self.model.predict(test_data)