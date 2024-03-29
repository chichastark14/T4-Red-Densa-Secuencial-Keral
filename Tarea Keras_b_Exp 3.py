#Se cambio la función de costo y se modifico las capas y neuronas
#No hubo mejora significativa con el primer commit pero en tiempo de ejecución si fue más lenta

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.datasets import mnist

# Carga de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocesamiento de datos
X_train = x_train[0:50000] / 255 
Y_train = keras.utils.to_categorical(y_train[0:50000], 10) 
X_val = x_train[50000:60000] / 255
Y_val = keras.utils.to_categorical(y_train[50000:60000], 10)
X_test = x_test / 255
Y_test = keras.utils.to_categorical(y_test, 10)

# Eliminar variables para liberar memoria
del x_train, y_train, x_test, y_test

# Reformatear datos para que sean adecuados para la red neuronal
X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
X_val = np.reshape(X_val, (X_val.shape[0], 28, 28, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
X_train_flat = X_train.reshape([X_train.shape[0], 784])
X_val_flat = X_val.reshape([X_val.shape[0], 784])
X_test_flat = X_test.reshape([X_test.shape[0], 784])

# Construcción del modelo
model = Sequential()
model.add(Dense(100, batch_input_shape=(None, 784)))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.add(Dense(30))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

# Compilación del modelo con función de pérdida 'mean_squared_error'
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(X_train_flat, Y_train, 
                    batch_size=128, 
                    epochs=90,
                    verbose=1, 
                    validation_data=(X_val_flat, Y_val))
