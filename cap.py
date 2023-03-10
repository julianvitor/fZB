import cv2
import numpy as np
import os
from PIL import Image
import pytesseract

# Abrindo o arquivo de vídeo
video = cv2.VideoCapture('video.mp4')

# Contador de frames e verificação se a leitura do vídeo foi bem sucedida
frame_count = 0
success = True

# Criação do diretório para armazenar os frames, caso ele não exista
try:
    if not os.path.exists('dados'):
        os.makedirs('dados')
except OSError:
    print('Erro: Não foi possível criar o diretório de dados')

while success:
    # Capturando o frame atual a cada 500ms
    video.set(cv2.CAP_PROP_POS_MSEC, (frame_count * 500))
    success, frame = video.read()

    # Nome do arquivo para salvar o frame
    nome_arquivo = f'./dados/frame{frame_count}.png'

    # Encerra o loop quando identifica o último frame
    if frame_count > 0:
        frame_anterior = cv2.imread(f'./dados/frame{frame_count - 1}.png')
        if np.array_equal(frame, frame_anterior):
            break

    # Salvando o frame como arquivo PNG
    cv2.imwrite(nome_arquivo, frame)
    print(f'Criando {nome_arquivo}')
    frame_count += 1

    # Ajuste de contraste e brilho da imagem
    contraste = float(3)
    brilho = int(100)
    img = cv2.imread(str(nome_arquivo))

    img_ajustada = cv2.multiply(img, np.array([contraste]))
    img_ajustada = cv2.add(img_ajustada, brilho)
    cv2.imwrite('./dados/contraste.png', img_ajustada)

    # Convertendo a imagem para o formato PIL
    img_pil = Image.open(nome_arquivo).convert('RGB')

    # Carregando a imagem ajustada de contraste e brilho como um array numpy
    img_ajustada = Image.open('./dados/contraste.png')
    np_img = np.asarray(img_ajustada).astype(np.uint8)

    # Aplicando limiarização na imagem em escala de cinza
    np_img[:, :, 0] = 0
    np_img[:, :, 2] = 0
    im_cinza = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(im_cinza, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh_adaptativa = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    bin_imagem = Image.fromarray(thresh_adaptativa)

    # Convertendo a imagem binarizada para texto utilizando OCR
    texto_imagem = pytesseract.image_to_string(bin_imagem)

    # Salvando o texto extraído em um arquivo de texto
    with open('./dados/dados.txt', 'a') as arquivo:
        arquivo.write(texto_imagem)

    # Excluindo os arquivos temporários
    os.remove("./dados/contraste.png")
    os.remove(nome_arquivo)

print('Extração de texto concluída')
