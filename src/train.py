import tensorflow as tf
import os
from config import *
from build_model import criar_modelo_base, criar_modelo_com_augmentation

def carregar_dados():
    treino_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    # Cache e prefetch para otimizar o uso da CPU/GPU
    AUTOTUNE = tf.data.AUTOTUNE
    treino_ds = treino_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return treino_ds, val_ds

def treinar_e_salvar(modelo, nome_modelo, treino_ds, val_ds):
    print(f"\n--- Treinando {nome_modelo} ---")
    history = modelo.fit(treino_ds, validation_data=val_ds, epochs=EPOCHS)
    
    caminho_salvar = os.path.join(MODEL_DIR, f"{nome_modelo}.keras")
    modelo.save(caminho_salvar)
    print(f"Modelo salvo em: {caminho_salvar}")
    return history

if __name__ == "__main__":
    treino_ds, val_ds = carregar_dados()
    input_shape = IMG_SIZE + (3,)
    
    # Treino Experimento 1
    modelo_base = criar_modelo_base(input_shape, NUM_CLASSES)
    treinar_e_salvar(modelo_base, "modelo_sem_aug", treino_ds, val_ds)
    
    # Treino Experimento 2
    modelo_aug = criar_modelo_com_augmentation(input_shape, NUM_CLASSES)
    treinar_e_salvar(modelo_aug, "modelo_com_aug", treino_ds, val_ds)