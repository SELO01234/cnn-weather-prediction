# EE beated it
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


#file_path = os.path.join(dest, "combined_data.npy") bunu değiştir path'e göre


os.path.exists(file_path):
    combined_data = np.load(file_path, allow_pickle=True).item()
    print("uploaded.")

if 'combined_data' in locals():
    separated_data = combined_data['separated_data']
    separated_labels = combined_data['separated_labels']

    train_data, train_labels = [], []
    val_data, val_labels = [], []
    test_data, test_labels = [], []

    for data, labels in zip(separated_data, separated_labels):
        num_samples = len(data)
        num_train = int(0.7 * num_samples)
        num_val = int(0.15 * num_samples)

        train_data.extend(data[:num_train])
        train_labels.extend(labels[:num_train])
        val_data.extend(data[num_train:num_train + num_val])
        val_labels.extend(labels[num_train:num_train + num_val])
        test_data.extend(data[num_train + num_val:])
        test_labels.extend(labels[num_train + num_val:])

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
   
    
    class ImageModel:

        def __init__(self):
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.model = None

        def build(self):
            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(100, 100, 3)))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(1024, activation='relu'))
            self.model.add(Dropout(0.5))  # Dropoutlu daha iyi
            self.model.add(Dense(4, activation='softmax'))

        def summary(self):
            self.model.summary()

        def compile(self):
            self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        def fit(self, train_data, train_labels, val_data, val_labels):
            
            self.model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=30, verbose=2)
          
            #model_save_path = ''
            self.model.save(model_save_path)
            print(f"Model saved to {model_save_path}")


        def predict(self, test_data):
            return self.model.predict(test_data)

    model = ImageModel()
    model.build()
    model.compile()
    model.summary()
    model.fit(train_data, train_labels, val_data, val_labels)

    
    loss, accuracy = model.model.evaluate(test_data, test_labels, verbose=2)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

