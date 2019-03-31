# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:13:26 2019
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist #28x28 tamaño de resolución 0-9
(x_entrenar, y_entrenar), (x_prueba, y_prueba) = mnist.load_data() 

x_entrenar = tf.keras.utils.normalize(x_entrenar, axis = 1) #Normalizar
x_prueba = tf.keras.utils.normalize(x_prueba, axis = 1) # datos de prueba y entrenamiento
 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) #Capas de entrada
model.add(tf.keras.layers.Dense(128, activacion = tf.nn.relu)) #Capa oculta con 128 neuronas en ella
model.add(tf.keras.layers.Dense(128, activacion = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activacion = tf.nn.softmax)) # 10 por ser del 0-9

model.compile(optimizar = 'adam', #adam es otra forma de optimizar, como descenso de gradiente 
              error = 'sparse_categorical_crossentropy', # margen de error 
              metrica = ['accuracy'])
model.fit(x_entrenar, y_entrenar, epochs = 3) #epochs son periodos para entrenar

val_error, val_acc = model.evaluar(x_prueba, y_prueba)
print(val_error, val_acc)

plt.imshow(x_entrenar[0], cmap = plt.cm.binary)
plt.show() # Muestra la imagen a colores
            #Para que sea blanco y negro --> cmap = plt.cm.binary
print(x_entrenar[0]) # Imprime el valor en tensor de la imagen --> Normalizar

model.save('epic_num_reader.model') #Guardar el midelo
new_model = tf.keras.models.load_model('epic_num_reader.model')
predicciones = new_model.predict([x_prueba])
print(predicciones)

print(np.argmax(predicciones[0])) # Muestra la predicción

plt.imshow(x_prueba[0])
plt.show()  #Muestra la imagen respectiva de la predicción


