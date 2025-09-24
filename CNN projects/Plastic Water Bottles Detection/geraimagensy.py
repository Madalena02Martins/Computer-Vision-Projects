from random import random, randint
from PIL import Image, ImageEnhance
import imutils
import os
import shutil

import cv2

def novofact(sit):
    x = random()
    if sit == 1  and x < 0.5 :
        x = x + 0.5
    elif sit == 2 and x > 0.5:
        x = x - 0.5
    if sit == 2:
        x = 1 + x
    return(x)

    

def gerabrilho(fiche,cami,ni):
    global indf


    nome, ext = os.path.splitext(fiche)
    im = Image.open(fiche)
    enhancer = ImageEnhance.Brightness(im)

    factor = 1 
    im_output = enhancer.enhance(factor)
    for i in range(0,ni):
        fat = novofact(1)
        im_output = enhancer.enhance(fat)
        fich = nome + "_" + str(indf) + ext
        im_output.save(fich,format="JPEG", quality=100)
        os.chdir("../labels")

        fichlab1=   nome + ".txt"
        fichlab2=   nome + "_" + str(indf) + ".txt"

        shutil.copy(fichlab1,fichlab2)
        os.chdir("../images")
        indf = indf+1

    for i in range(0,ni):
        fat = novofact(2)
        im_output1 = enhancer.enhance(fat)
        fich = nome + "_" + str(indf) + ext
        im_output.save(fich,format="JPEG", quality=100)
        os.chdir("../labels")

        fichlab1=   nome + ".txt"
        fichlab2=   nome + "_" + str(indf) + ".txt"
        shutil.copy(fichlab1,fichlab2)
        os.chdir("../images")
        indf = indf+1



def gerarotacao(fiche,cami,ni):
    global indf

    nome, ext = os.path.splitext(fiche)
    im = cv2.imread(fiche)
    for i in range(0,ni):
        vr = randint(0, 10)
        imrod = imutils.rotate(im, angle=vr) 
        fich = nome + "_" + str(indf) + ext
        cv2.imwrite(fich,imrod)
        os.chdir("../labels")
        fichlab1=   nome + ".txt"
        fichlab2=   nome + "_" + str(indf) + ".txt"
        shutil.copy(fichlab1,fichlab2)
        os.chdir("../images")
        indf = indf + 1

    for i in range(0, ni):
        vr = (-1)*randint(0, 10)
        imrod = imutils.rotate(im, angle=vr) 
        fich = nome + "_" + str(indf) + ext
        cv2.imwrite(fich,imrod)
        os.chdir("../labels")
        fichlab1=   nome + ".txt"
        fichlab2=  nome + "_" + str(indf) + ".txt"
        shutil.copy(fichlab1,fichlab2)
        os.chdir("../images")
        indf = indf + 1

def gerar_novas(campa):
    global indf
    
    lpa = ["images"]
    indf = 0
    for p in lpa:
        os.chdir(campa)
        os.chdir(p)
        lf = os.listdir() #cam

        for f in lf:
            if 'jpg' in f:
                gerarotacao(f,campa,2)
            
        lf = os.listdir()  #cam
        for f in lf:
            if 'jpg' in f:
                gerabrilho(f,campa,3)


#-------------------------------------------------------------------------------------------------    
def muda_nome(camp):
    os. chdir(camp)
    lf = os.listdir(camp) 

    for f in lf:
        nome, ext = os.path.splitext(f)
        if '-' in nome:
            pos = nome.find('-')
            nnome = nome[pos+1:len(nome)]
            os.rename(f, nnome +  ext)

gerar_novas("C:\\Users\\madal\\OneDrive\\Ambiente de Trabalho\\GitHub Computer Vision\\OpenCV Projects\\CNN projects\\Plastic Water Bottles Detection\\train")


