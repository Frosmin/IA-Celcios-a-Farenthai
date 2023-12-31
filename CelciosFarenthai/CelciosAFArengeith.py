import tensorflow as tf 
import numpy as np

celsius = np.array([-40,-10,0,8,15,22,38], dtype = float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype = float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
      optimizer=tf.keras.optimizers.Adam(0.1),
      loss='mean_squared_error'
)

print("comienza el entrenamiento")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("modelo entrenado :D")

import matplotlib.pyplot as plt 
plt.xlabel("# epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

print("hagamos una predicion")
resultado = modelo.predict([100.0])
print(str(resultado))