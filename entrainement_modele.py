import matplotlib.pyplot as plt
import numpy as np
import os
import random
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from collections import defaultdict
# print(K.image_data_format)
import unet
import tensorflow as tf
import h5py
# import resnet
# from tf.python.ops.clip_ops import clip_by_value

def plot_some_results(data, target, img_sufixe):
    """__author__ = 'Fabian Isensee'
    https://github.com/Lasagne/Recips/blob/master/examples/UNet/massachusetts_road_segm.py"""
    d = data
    s = target
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(d.transpose(1,2,0))
    plt.title("input patch")
    plt.subplot(1, 3, 2)
    plt.imshow(s[0])
    plt.title("ground truth")
    plt.savefig(os.path.join("D:\Processus\image_to_echantillons\img_1", "result_%03.0f.png"%img_sufixe))
    plt.close()

def EncoderLabels(input_batch, num_classes):
    # One-hot encoding for reference map
    # print(input_batch.shape)
    batch_size, etiquette, n_rows, n_cols = input_batch.shape
    x1d = input_batch.ravel()
    # print(x1d)
    y1d = to_categorical( x1d, num_classes = num_classes )
    y4d = y1d.reshape( [ batch_size, num_classes, n_rows, n_cols ] )
    
    return y4d

def ChargerDonnees(DossierEchantillons, TypeEchantillons, nbreEchantillons, Taille_tuile):
    """ 
    Code modifie de https://stackoverflow.com/questions/29380403/reading-a-binary-file-into-2d-array-python
    en combinaison avec
    https://stackoverflow.com/questions/30124255/read-a-binary-file-using-numpy-fromfile-and-a-given-offset
    """
    # TypeEchantillons = "validation" ou "entrainement"
    
    # fichier_info = open(os.path.join(DossierEchantillons, 'info.txt'), 'r')

    # liste pour randomizer les echantillons
    # listRdm = range(0, nbreEchantillons)
    listRdm = []
    for i in range(0, nbreEchantillons):
        listRdm.append(i)
    random.shuffle(listRdm)
    # print(listRdm)
 
    # tailleTuile = 512
    data = np.empty([nbreEchantillons, 3, Taille_tuile, Taille_tuile], dtype=np.uint8)
    target = np.empty([nbreEchantillons, 1, Taille_tuile, Taille_tuile], dtype=np.uint8)
    # matrices = np.load(os.path.join(DossierEchantillons, "echantillons_"+TypeEchantillons, "echantillons_RGB.npy"))
    for i in range (0, nbreEchantillons):
        indiceData = listRdm[i]
        
        data_file = open(os.path.join(DossierEchantillons, "echantillons_"+TypeEchantillons, "echantillons_RGB.dat"), "rb")
        ref_file = open(os.path.join(DossierEchantillons, "echantillons_"+TypeEchantillons, "echantillons_Label.dat"), "rb")   
        data_file.seek(i*Taille_tuile*Taille_tuile*3)
        ref_file.seek(i*Taille_tuile*Taille_tuile)
        
        data[indiceData, :, :, :] = np.reshape(np.fromfile(data_file, dtype=np.uint8, count=3*Taille_tuile*Taille_tuile), [3, Taille_tuile, Taille_tuile]) 
        target[indiceData, :, :, :] = np.reshape(np.fromfile(ref_file, dtype=np.uint8, count=Taille_tuile*Taille_tuile), [1, Taille_tuile, Taille_tuile])
        
        d = np.reshape(np.fromfile(data_file, dtype=np.uint8, count=3*Taille_tuile*Taille_tuile), [3, Taille_tuile, Taille_tuile]) 
        t = np.reshape(np.fromfile(ref_file, dtype=np.uint8, count=Taille_tuile*Taille_tuile), [1, Taille_tuile, Taille_tuile])
        
        # plot_some_results(d, t, i)
    data_file.close()
    ref_file.close()
    
    return data, target

def train_net(TravailFolder, pretrained, TailleBatch, NbEpoques, TailleTuile, NbClasses, tache="segmentation"):
    if tache == "segmentation":
        x_val, y_val = ChargerDonnees(TravailFolder, "validation", 2, Taille_tuile)
        x_trn, y_trn = ChargerDonnees(TravailFolder, "entrainement", 50, Taille_tuile)
        
        y_trn= EncoderLabels(y_trn, num_classes=NbClasses)
        y_val= EncoderLabels(y_val, num_classes=NbClasses)
        # print(y_trn.shape)
        
        
        # Preparer l'augmentation de donnees a la volee
        datagen = ImageDataGenerator(featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True,channel_shift_range=10.)
        
        # model = unet.get_unet(TailleTuile,NbClasses)
        model = unet.get_unet_Conv_Block(TailleTuile,NbClasses)
    
        # Load pre-trained model?
        if pretrained:
            model.load_weights('filepath_vers_poids.hdf5')
        
        model_checkpoint = ModelCheckpoint(os.path.join(TravailFolder, 'unet_tmp.hdf5'), monitor='loss', save_best_only=True)
        
        # Fit le model sur les donnees d'entrainement et fait l'augmentation de donnees a la volee.
        datagen.fit(x_trn)
        model.fit_generator(datagen.flow(x_trn, y_trn, batch_size=TailleBatch), steps_per_epoch=len(x_trn) / TailleBatch, epochs=NbEpoques, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_val, y_val))
        
        # model.fit(x_trn, y_trn, batch_size=TailleBatch, epochs=NbEpoques, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_val, y_val)) 
        del x_trn
        del y_trn
        model.save_weights(os.path.join(TravailFolder, "unet_final.hdf5"))
        
    # elif tache == "classification":
        # Charger le Cifar10 dataset
        # from keras.datasets import cifar10
        # (x_trn, y_trn), (x_val, y_val) = cifar10.load_data()
        
        # x_trn = x_trn.astypt('float32')        
        
        # model = resnet.ResNet50(input_shape=)

### parametres
Taille_batch = 2
Nbre_epoque = 1
Taille_tuile = 512
Nbre_classes = 4
# TravailDossier = "/gpfs/fs1/nrcan/nrcan_geobase/extraction/Deep_learning/tensorflowtest"
TravailDossier = "D:\Processus\image_to_echantillons\img_1"
pretrn = False

train_net(TravailDossier, pretrn, Taille_batch, Nbre_epoque, Taille_tuile, Nbre_classes)        

        