import os
from PIL import Image
from config import TEST_DIR

print("Verificando imagens em:", TEST_DIR)

for root, dirs, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pgn')):
            caminho_arquivo = os.path.join(root, file)
            try:
                imagem = Image.open(caminho_arquivo)
                # Se a imagem tiver transparência (RGBA) ou for tons de cinza (L)
                if imagem.mode != 'RGB':
                    # Converte forçadamente para RGB (3 canais)
                    imagem = imagem.convert('RGB')
                    imagem.save(caminho_arquivo)
                    print(f"Corrigida: {file}")
            except Exception as e:
                print(f"Aviso: Não foi possível processar {file} - {e}")

print("Verificação concluída!")