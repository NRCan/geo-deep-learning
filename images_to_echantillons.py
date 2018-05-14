import numpy as np
import os
from PIL import Image
import fnmatch
import matplotlib.pyplot as plt
import random

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
    
# fonction de preparation des echantillons, a partir des listes d'images RGB et Label.
def PrepEchantillons(ImagesRGB, ImagesLabel, ImagesFolder, OutputFolder, tailleTuile, chevauchement):
    # verifier que la liste des images rgb est de la meme taille que la liste des etiquettes.
    assert(len(ImagesRGB) == len(ImagesLabel))
    # compter le nombre d'echantillons ecrit
    compteur = 0
    
    for img in ImagesRGB:
        print(os.path.join(ImagesFolder, "RGB", img))
        
        # lecture des images rgb et label comme des matrices
        RGBArray = np.array(Image.open(os.path.join(ImagesFolder, "RGB", img)))
        hauteur, largeur, nbbande = RGBArray.shape
        LabelArray = np.array(Image.open(os.path.join(ImagesFolder, "label", ImagesLabel[ImagesRGB.index(img)])))     
        
        # zero padding autour des images. taille du padding egale a une demi-tuile.
        transp = np.transpose(RGBArray, (2, 0, 1))
        pad_RGB_Array = np.pad(transp, ((0,0),(int(tailleTuile / 2), int(tailleTuile / 2)),(int(tailleTuile / 2), int(tailleTuile / 2))), mode='constant')
        pad_Label_Array = np.pad(LabelArray, ((int(tailleTuile / 2), int(tailleTuile / 2)),(int(tailleTuile / 2), int(tailleTuile / 2))), mode='constant')
        
        # print(pad_Label_Array.dtype)
        for row in range(0, largeur, chevauchement):
            for column in range(0, hauteur, chevauchement):
                # on ecrit dans le .dat la portion de l image correspondant a la tuile
                data = (pad_RGB_Array[:,row:row+tailleTuile, column:column+tailleTuile])
                target = (pad_Label_Array[row:row+tailleTuile, column:column+tailleTuile])
                
                # print(data.shape)                
                # plot_some_results(data, target, compteur, OutputFolder)
                
                nomEchantRGB = os.path.join(OutputFolder, "echant_RGB", str(img + "_" + str(compteur) + ".tif"))
                nomEchantLabel = os.path.join(OutputFolder, "echant_label", str(ImagesLabel[ImagesRGB.index(img)] + "_" + str(compteur) + ".tif"))
                
                with open(os.path.join(OutputFolder, "info_img.txt"), 'a') as the_file:
                    the_file.write(nomEchantRGB + " " + nomEchantLabel + " " + str(compteur) + "\n")
                
                compteur+=1
                
                ecrireArray(os.path.join(OutputFolder, "tmp_echantillons_RGB.dat"), data)
                ecrireArray(os.path.join(OutputFolder, "tmp_echantillons_Label.dat"), target)
        
        
                # print "max: " + str(max(target.ravel())) + ", min: " + str(min(target.ravel()))
    return compteur

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


### Debut des traitements ###

# parametres
# TODO lire les parametres dans un fichier txt.
images_Folder =  'D:/Processus/image_to_echantillons/img_1'
path_Echantillons_training = 'D:/Processus/image_to_echantillons/img_1/echantillons_entrainement'
# images_Folder =  '/gpfs/fs1/nrcan/nrcan_geobase/extraction/Deep_learning/pytorch'
# path_Echantillons_training = '/gpfs/fs1/nrcan/nrcan_geobase/extraction/Deep_learning/pytorch/echantillons_entrainement'
largeur_tuile = 256
# chevauchement = espace entre 2 centres de tuiles
chevauchement = 150
 
# lister les images dans les dossiers RGB et Label
images_RGB = [img for img in os.listdir(os.path.join(images_Folder, "RGB")) if fnmatch.fnmatch(img, "*.tif*")]
images_Label = [img for img in os.listdir(os.path.join(images_Folder, "label")) if fnmatch.fnmatch(img, "*.tif*")]
images_RGB.sort()
images_Label.sort()
print(images_RGB)
print(images_Label)
 
# Ecrire les echantillons dans 2 fichiers .dat 
nbrEchant = PrepEchantillons(images_RGB, images_Label, images_Folder, path_Echantillons_training, largeur_tuile, chevauchement)
randomisationEchantillons(path_Echantillons_training, largeur_tuile, chevauchement, nbrEchant)
print(nbrEchant)
# TODO ecrire le nombre d'echantillons dans un fichier txt.
# TODO ecrire le nombre de classes des donnees de ref dans un fichier txt.
         
