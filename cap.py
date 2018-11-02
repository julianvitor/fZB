import cv2
import numpy as np
import os
from PIL import Image  # Importando o módulo Pillow para abrir a imagem no script
import pytesseract  # Módulo para a utilização da tecnologia OCR



vidcap = cv2.VideoCapture('video.mp4')
count = 0
success = True
try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print('Error: Creating directory of data')




while success:
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*500))#controla a quantidade quadros, 1000 = 1 quadro a cada 1000ms
    success,frame = vidcap.read()

    name = './data/frame' + str(count) + '.png'

    # Encerra quando identifica o ultimo frame
    image_last = cv2.imread("frame{}.png".format(count-1))
    if np.array_equal(frame,image_last):
        break

    cv2.imwrite(name, frame)     # save frame as PNG file
    print('Creating...' + name)
    count += 1







########################################################################33




    alpha = float(3)  # Controle de contraste 1-3
    beta = int(100)  # Controle de brilho 0 -100

    img = cv2.imread(str(name))

    mul_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
    new_img = cv2.add(mul_img, beta)  # new_img = img*alpha + beta
    cv2.imwrite('./data/contraste.png', new_img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter cor para pil
    img_pil = Image.fromarray(img)  # Converter para pil

    img = Image.open('./data/contraste.png')

    # convertendo em um array editável de numpy[x, y, CANALS]
    npimg = np.asarray(img).astype(np.uint8)

    # diminuição dos ruidos antes da binarização
    npimg[:, :, 0] = 0  # zerando o canal R (RED)
    npimg[:, :, 2] = 0  # zerando o canal B (BLUE)

    # atribuição em escala de cinza
    im = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)

    # aplicação da truncagem binária para a intensidade
    # pixels de intensidade de cor abaixo de 127 serão convertidos para 0 (PRETO)
    # pixels de intensidade de cor acima de 127 serão convertidos para 255 (BRANCO)
    # A atrubição do THRESH_OTSU incrementa uma analise inteligente dos nivels de truncagem
    ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    th3 = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # reconvertendo o retorno do threshold em um objeto do tipo PIL.Image
    binimagem = Image.fromarray(th3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    its = pytesseract.image_to_string(binimagem)  # Extraindo o texto da imagem

    file = open('./data/data.txt', 'r') #criando arquivo de texto para salvar o que foi lido

    content = file.readlines()
    content.append(its)
    file = open('./data/data.txt', 'w')
    file.writelines(content)

    os.remove("./data/contraste.png")
    os.remove(name)
    file.close()


