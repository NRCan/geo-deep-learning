import numpy as np
import os
from PIL import Image
import fnmatch
import matplotlib.pyplot as plt
import random
import argparse

def plot_some_results(data, target, img_sufixe, dossierTravail):
    """__author__ = 'Fabian Isensee'
    https://github.com/Lasagne/Recips/blob/master/examples/UNet/massachusetts_road_segm.py"""
    d = data
    s = target
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(d.transpose(1,2,0))
    plt.title("input patch")
    plt.subplot(1, 3, 2)
    plt.imshow(s)
    plt.title("ground truth")
    plt.savefig(os.path.join(dossierTravail, "result_%03.0f.png"%img_sufixe))
    plt.close()

# fonction pour ecrire en mode append les matrices 
def ecrireArray(fichier, matrice):
    with open(fichier, 'ab') as out_file:
    # out_file = open(fichier, 'ab')
        matrice.tofile(out_file)
    
def rdmList(nbrEchant):
    listRdm = []
    for i in range(nbrEchant):
        listRdm.append(i)
    random.shuffle(listRdm)
    return listRdm

def ReadParameters(ParamFile):
    with open(ParamFile) as f:
        lines = f.readlines()
    content = [x.strip() for x in lines]
    image_folder = content[0]
    path_Echantillons = content[1]
    largeur_tuile = int(content[2])
    chevauchement = int(content[3])
    return image_folder, path_Echantillons, largeur_tuile, chevauchement
    
def WriteInfo(OutputFolder, nbEchant, nbClasses):
    with open(os.path.join(OutputFolder, "info.txt"), 'a') as the_file:
        the_file.write(str(nbEchant) + "\n")
        the_file.write(str(nbClasses + 1) + "\n")
    the_file.close()
    
# fonction de preparation des echantillons, a partir des listes d'images RGB et Label.
def PrepEchantillons(ImagesRGB, ImagesLabel, ImagesFolder, OutputFolder, tailleTuile, chevauchement):
    # verifier que la liste des images rgb est de la meme taille que la liste des etiquettes.
    assert(len(ImagesRGB) == len(ImagesLabel))
    # compter le nombre d'echantillons ecrit
    compteur = 0
    num_classes = 0
    for img in ImagesRGB:
        print(os.path.join(ImagesFolder, "RGB", img))
        
        # lecture des images rgb et label comme des matrices
        RGBArray = np.array(Image.open(os.path.join(ImagesFolder, "RGB", img)))
        hauteur, largeur, nbbande = RGBArray.shape
        LabelArray = np.array(Image.open(os.path.join(ImagesFolder, "label", ImagesLabel[ImagesRGB.index(img)])))     
        
        # zero padding autour des images. taille du padding egale a une demi-tuile.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        transp = np.transpose(RGBArray, (2, 0, 1))
        pad_RGB_Array = np.pad(transp, ((0,0),(int(tailleTuile / 2), int(tailleTuile / 2)),(int(tailleTuile / 2), int(tailleTuile / 2))), mode='constant')
        pad_Label_Array = np.pad(LabelArray, ((int(tailleTuile / 2), int(tailleTuile / 2)),(int(tailleTuile / 2), int(tailleTuile / 2))), mode='constant')
        
        # print(pad_Label_Array.dtype)
        for row in range(0, largeur, chevauchement):
            for column in range(0, hauteur, chevauchement):
                # on ecrit dans le .dat la portion de l image correspondant a la tuile
                data = (pad_RGB_Array[:,row:row+tailleTuile, column:column+tailleTuile])
                target = (pad_Label_Array[row:row+tailleTuile, column:column+tailleTuile])
                
                compteur+=1
                # Loading array is much more efficient than images.
                ecrireArray(os.path.join(OutputFolder, "tmp_echantillons_RGB.dat"), data)
                ecrireArray(os.path.join(OutputFolder, "tmp_echantillons_Label.dat"), target)
                
                if num_classes < max(target.ravel()):
                    num_classes = max(target.ravel())
    return compteur, num_classes

def randomisationEchantillons(OutputFolder, tailleTuile, chevauchement, Nbrechant):
    RdmEchant = rdmList(Nbrechant)
    
    for idx in RdmEchant:
        data_file = open(os.path.join(OutputFolder, "tmp_echantillons_RGB.dat"), "rb")
        ref_file = open(os.path.join(OutputFolder, "tmp_echantillons_Label.dat"), "rb")   
        data_file.seek(idx*tailleTuile*tailleTuile*3)
        ref_file.seek(idx*tailleTuile*tailleTuile)
        
        data = np.fromfile(data_file, dtype=np.uint8, count=3*tailleTuile*tailleTuile)
        target = np.fromfile(ref_file, dtype=np.uint8, count=tailleTuile*tailleTuile)
        
        ecrireArray(os.path.join(OutputFolder, "echantillons_RGB.dat"), data)
        ecrireArray(os.path.join(OutputFolder, "echantillons_Label.dat"), target)
    
    # Delete temp files.
    data_file.close()
    ref_file.close()
    os.remove(os.path.join(OutputFolder, "tmp_echantillons_RGB.dat"))
    os.remove(os.path.join(OutputFolder, "tmp_echantillons_Label.dat"))

### Debut des traitements ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample preparation')
    parser.add_argument('ParamFile', metavar='DIR',
                        help='path to parameters txt')
    args = parser.parse_args()
    images_Folder, path_Echantillons, largeur_tuile, chevauchement = ReadParameters(args.ParamFile)
    
    # lister les images dans les dossiers RGB et Label
    images_RGB = [img for img in os.listdir(os.path.join(images_Folder, "RGB")) if fnmatch.fnmatch(img, "*.tif*")]
    images_Label = [img for img in os.listdir(os.path.join(images_Folder, "label")) if fnmatch.fnmatch(img, "*.tif*")]
    images_RGB.sort()
    images_Label.sort()
     
    # Ecrire les echantillons dans 2 fichiers .dat 
    nbrEchant, nbrClasses = PrepEchantillons(images_RGB, images_Label, images_Folder, path_Echantillons, largeur_tuile, chevauchement)
    randomisationEchantillons(path_Echantillons, largeur_tuile, chevauchement, nbrEchant)
    WriteInfo(path_Echantillons, nbrEchant, nbrClasses)
    print("Terminado")
        
