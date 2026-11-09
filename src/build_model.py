import tensorflow as tf
from tensorflow.keras import layers, models

def criar_modelo_base(input_shape, num_classes):
    modelo = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Rescaling(1./255), # Normalização dos pixels para [0, 1]
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

def criar_modelo_com_augmentation(input_shape, num_classes):
    # Transformações aplicadas apenas durante o treinamento
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2)
    ])
    
    modelo = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        data_augmentation,
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo