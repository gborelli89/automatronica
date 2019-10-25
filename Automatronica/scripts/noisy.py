import cv2
import numpy as np
import os

# image: imagem 
# amount: porcentagem de ruído (entre 0 e 1)
# s_vs_p: razão entre salt e pepper (entre 0 e 1)
def sp_noisy(image, amount, s_vs_p):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row, col = imgray.shape
    
    out = imgray.copy()
    # Salt mode
    num_salt = np.ceil(amount * imgray.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in imgray.shape]
    out[coords] = 255

    # Pepper mode
    num_pepper = np.ceil(amount* imgray.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in imgray.shape]
    out[coords] = 0
    return out

# image: imagem 
# mean: media do ruído gaussiano (considerar 0)
# sigma: desvio padrão da distribuição gaussiana (valor inteiro)
def gauss_noisy(image, mean, sigma):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row,col= imgray.shape

    gauss = np.random.normal(mean,sigma,(row,col)).astype(np.uint8)
    #gauss = gauss.reshape(row,col,ch)
    noisy = np.clip(imgray + gauss, 0, 255)
    return noisy
    
def speckle_noisy(image):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row, col = imgray.shape
    
    gauss = np.random.randn(row,col)
    gauss = gauss.reshape(row,col)
    alpha = imgray * gauss
    noisy = np.clip(imgray + alpha.astype(np.uint8), 0, 255)
    return noisy