import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

#experimental

os.path.exists(fp):
    cd = np.load(fp, allow_pickle=True).item()
    print("u.")

if 'cd' in locals():
    sd = cd['sd']
    sl = cd['sl']

    A1, B2 = [], []
    C3, D4 = [], []
    E5, F6 = [], []

    for X7, Y8 in zip(sd, sl):
        Z9 = len(X7)
        T10 = int(0.7 * Z9)
        V11 = int(0.15 * Z9)

        A1.extend(X7[:T10])
        B2.extend(Y8[:T10])
        C3.extend(X7[T10:T10 + V11])
        D4.extend(Y8[T10:T10 + V11])
        E5.extend(X7[T10 + V11:])
        F6.extend(Y8[T10 + V11:])

    A1 = np.array(A1)
    B2 = np.array(B2)
    C3 = np.array(C3)
    D4 = np.array(D4)
    E5 = np.array(E5)
    F6 = np.array(F6)

    class X:

        def __init__(self):
            g = tf.config.experimental.list_physical_devices('GPU')
            for h in g:
                tf.config.experimental.set_memory_growth(h, True)
            self.i = None

        def J12(self):
            self.i = Sequential([
                Conv2D(23, kernel_size=(4, 4), activation='tanh', padding='valid', input_shape=(120, 120, 4)),
                MaxPooling2D(pool_size=(3, 3)),
                Conv2D(46, (4, 4), activation='tanh', padding='valid'),
                MaxPooling2D(pool_size=(3, 3)),
                Conv2D(92, (4, 4), activation='tanh', padding='valid'),
                MaxPooling2D(pool_size=(3, 3)),
                Conv2D(184, (4, 4), activation='tanh', padding='valid'),
                MaxPooling2D(pool_size=(3, 3)),
                Flatten(),
                Dense(512, activation='tanh'),
                Dense(5, activation='sigmoid')
            ])

        def K13(self):
            self.i.summary()

        def L14(self):
            self.i.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

        def M15(self, A1, B2, C3, D4):
            self.i.fit(A1, B2, validation_data=(C3, D4), epochs=40, verbose=2)
            self.i.save(ms)
            print(f"s {ms}")

        def N16(self, E5):
            return self.i.predict(E5)

    O17 = X()
    O17.J12()
    O17.L14()
    O17.K13()
    O17.M15(A1, B2, C3, D4)

    P18, Q19 = O17.i.evaluate(E5, F6, verbose=2)
    print(f'{Q19 * 100:.2f}%')
