# src/config.py
import os

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 6

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'treino')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'minhas_fotos')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')