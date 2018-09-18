import cv2
import numpy as np
import os
from PIL import Image # Importando o módulo Pillow para abrir a imagem no script
import pytesseract # Módulo para a utilização da tecnologia OCR
import math

def vtf ():
    
    # Playing video from file:
    cap = cv2.VideoCapture('video3.mp4')
   

    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print ('Error: Creating directory of data')
    ret = True
    currentFrame = 0


    while(ret):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Saves image of the current frame in png file
        name = './data/frame' + str(currentFrame) + '.png'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def rec ():


    alpha = float(3)     # Simple contrast control 1-3
    beta = int(100)             # Simple brightness control 0 -100
     

    img = cv2.imread('imagem4.jpg')
     
    mul_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
    new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta
    cv2.imwrite('contraste.png', new_img)



    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert color for pil
    img_pil = Image.fromarray(img) #convert fo pil
    




    


    img = Image.open('contraste.png')

    # convertendo em um array editável de numpy[x, y, CANALS]
    npimg = np.asarray(img).astype(np.uint8) 


    # diminuição dos ruidos antes da binarização
    npimg[:, :, 0] = 0 # zerando o canal R (RED)
    npimg[:, :, 2] = 0 # zerando o canal B (BLUE)

    # atribuição em escala de cinza
    im = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)

    # aplicação da truncagem binária para a intensidade
    # pixels de intensidade de cor abaixo de 127 serão convertidos para 0 (PRETO)
    # pixels de intensidade de cor acima de 127 serão convertidos para 255 (BRANCO)
    # A atrubição do THRESH_OTSU incrementa uma análise inteligente dos nivels de truncagem
    ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    th3 = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # reconvertendo o retorno do threshold em um objeto do tipo PIL.Image
    binimagem = Image.fromarray(th3) 

    os.remove("contraste.png")

    

    cv2.waitKey(0)
    cv2.destroyAllWindows()









    
    
    its = pytesseract.image_to_string(binimagem, lang='por')  # Extraindo o texto da imagem

    cv2.imwrite('imagem4pp.png', th3)
    
    
    print(its)








#vtf()

rec()







