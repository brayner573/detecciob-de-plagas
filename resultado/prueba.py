import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Directorio de imágenes
dataset_dir = 'dataset'  # Asegúrate de que este sea el camino correcto a tu carpeta dataset

# Verificar el contenido de las carpetas
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    if os.path.isdir(folder_path):
        print(f"Contenido de '{folder}': {os.listdir(folder_path)}")

# Parámetros de la imagen
image_size = (128, 128)  # Tamaño de las imágenes
batch_size = 2  # Tamaño del lote reducido para evitar problemas con pocas imágenes

# Generador de datos con aumento de imagen
data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,  # Aumento de datos
    rotation_range=40
)

# Cargar imágenes de entrenamiento (sin validación)
train_data = data_gen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  # Multiclase, asegura etiquetas en formato one-hot
)

# Verificación de las clases encontradas
print(f"Clases en el conjunto de entrenamiento: {train_data.class_indices}")

# Si todo está bien, entrenar el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Cambiado para 2 clases: 'plaga' y 'no_plaga'
])

# Compilar el modelo con ejecución en modo eager
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

# Entrenar el modelo
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    epochs=10
)

# Guardar el modelo
model.save('modelo_plagas.h5')

# Graficar los resultados
plt.figure(figsize=(12, 4))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.title('Precisión del Modelo')
plt.xlabel('Epochs')
plt.ylabel('Precisión')
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.title('Pérdida del Modelo')
plt.xlabel('Epochs')
plt.ylabel('Pérdida')
plt.legend()

# Guardar los gráficos
plt.savefig('resultado/precision_perdida2.png')

# Mostrar los gráficos
plt.show()
