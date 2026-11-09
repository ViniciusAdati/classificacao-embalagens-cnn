import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from config import *

def predizer_e_avaliar(nome_modelo):
    caminho_modelo = os.path.join(MODEL_DIR, f"{nome_modelo}.keras")
    modelo = tf.keras.models.load_model(caminho_modelo)
    
    # Carrega as 10 fotos de teste
    teste_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False # Importante manter False para a matriz de confusão bater com as labels
    )
    
    nomes_classes = teste_ds.class_names
    y_true = np.concatenate([y for x, y in teste_ds], axis=0)
    
    predicoes = modelo.predict(teste_ds)
    y_pred = np.argmax(predicoes, axis=1)
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(nomes_classes)))
    
    # Plota a Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    
    caminho_grafico = os.path.join(RESULTS_DIR, f"matriz_confusao_{nome_modelo}.png")
    plt.savefig(caminho_grafico)
    print(f"Matriz de confusão salva em: {caminho_grafico}")
    plt.show()

if __name__ == "__main__":
    predizer_e_avaliar("modelo_sem_aug")
    predizer_e_avaliar("modelo_com_aug")