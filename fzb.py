import cv2
import numpy as np
import os
from PIL import Image
import pytesseract


def vtf(video_path):
    # Carrega o vídeo
    cap = cv2.VideoCapture(video_path)

    # Cria um diretório temporário para armazenar os frames
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print('Error: creating directory of data')
    
    # Loop para processar cada frame do vídeo
    ret = True
    currentFrame = 0

    while ret:
        # Captura o frame atual
        ret, frame = cap.read()

        if ret:
            # Salva a imagem do frame atual em um arquivo PNG
            name = f'./data/frame{currentFrame}.png'
            print(f'Creating {name}...')
            cv2.imwrite(name, frame)
            currentFrame += 1

    # Libera a captura de vídeo e destrói as janelas abertas pelo OpenCV
    cap.release()
    cv2.destroyAllWindows()


def rec(frame_path):
    # Ajusta o contraste e o brilho da imagem
    alpha = 3.0  # Fator de controle do contraste (1.0 = sem alteração)
    beta = 100  # Fator de controle do brilho (0 = preto, 100 = original)
    img = cv2.imread(frame_path)
    img_contrast = cv2.addWeighted(img, alpha, np.zeros_like(img), 0, beta)

    # Converte a imagem para escala de cinza
    img_gray = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2GRAY)

    # Aplica a binarização adaptativa
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Converte a imagem para o formato PIL.Image
    binimagem = Image.fromarray(thresh)

    # Remove o arquivo de contraste
    os.remove(frame_path)

    # Extrai o texto da imagem usando OCR
    its = pytesseract.image_to_string(binimagem)

    # Salva a imagem binarizada em um arquivo PNG
    cv2.imwrite('imagem4pp.png', thresh)

    print(its)


def main(video_path):
    # Gera os frames do vídeo
    vtf(video_path)

    # Processa cada frame e extrai o texto
    for i in range(len(os.listdir('data'))):
        frame_path = f'data/frame{i}.png'
        rec(frame_path)

    # Remove os arquivos temporários
    for file in os.listdir('data'):
        os.remove(os.path.join('data', file))
    os.rmdir('data')


if __name__ == '__main__':
    main('video.mp4')