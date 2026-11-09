# 📦 Classificação de Tipos de Embalagens com CNN e Data Augmentation

Este projeto utiliza Visão Computacional e Redes Neurais Convolucionais (CNNs) com TensorFlow/Keras para classificar diferentes tipos de embalagens e recipientes (Lata de Alumínio, Garrafa PET, Caixa de Papelão, Frasco de Vidro e Pote Plástico).

O objetivo principal do estudo é analisar o impacto de reflexos de luz, materiais transparentes e rótulos coloridos na predição da rede, além de comprovar a eficácia da técnica de **Data Augmentation** na prevenção de _overfitting_.

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python
- **Machine Learning:** TensorFlow / Keras, Scikit-Learn
- **Visão Computacional & Imagens:** Pillow (PIL)
- **Visualização de Dados:** Matplotlib, Seaborn

## 📂 Estrutura do Projeto

- `src/`: Scripts Python contendo a arquitetura da CNN, pipeline de treino e avaliação.
- `dataset/minhas_fotos/`: Fotos reais utilizadas para testar o modelo em ambientes do cotidiano (iluminações variadas, reflexos e rótulos).
- `results/`: Matrizes de Confusão geradas comparando as predições.

## 🔬 Experimentos e Resultados

Foram treinados dois modelos idênticos, variando apenas o pré-processamento dos dados:

1. **Modelo sem Augmentation:** Sofreu overfitting com as imagens de treino originais, apresentando dificuldades ao lidar com reflexos em embalagens metálicas e vidros transparentes, além de focar em rótulos coloridos ao invés do formato do recipiente plástico.
2. **Modelo com Data Augmentation:** Utilizou variações de rotação, contraste e brilho durante o treinamento, forçando a rede a aprender características estruturais robustas (bordas e formas), melhorando a generalização em fotos reais.

## 🚀 Como executar

1. Clone o repositório.
2. Crie e ative um ambiente virtual (`python -m venv venv`).
3. Instale as dependências: `pip install -r requirements.txt`.
4. Rode a avaliação para gerar as matrizes de confusão: `python src/evaluate.py`.
