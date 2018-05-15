# Projet d'apprentissage profond du CCCOT.

## Fichier images_to_echantillons.py  
Pour lancer le programme:  
``` 
python images_to_echantillons.py path/to/parameter_file.txt
```

Le fichier de paramètres contient les informations suivantes:  
```
path/to/folder/images_et_reference
path/to/folder/ecriture_des_echantillons
taille des tuiles
espacement entre centres de tuiles
```

Intrants: 
- Images RGB et référence
- Dossier de travail
- Taille des échantillons (tuiles) à créer, en pixel
- Espace entre 2 centres de tuiles (chevauchement), en pixel

Extrants:
- 2 fichiers .dat contenant les matrices (numpy.array) des tuiles créées.
- 1 fichier .txt contenant le nombre d'echantillons et le nombre de classe des données de ref.

Processus: 
- Lire les images dans le dossier "RGB" et dans le dossier "label"
- Convertir les images en matrices
- Subdiviser les images en échantillons de taille et d'espacement spécifié en intrant
- Écrire les échantillons dans 2 fichiers .dat (RGB et Label)

## Fichier entrainement_modele.py

Intrants:
- 2 fichiers .dat contenant les matrices RGB et Label pour l'entrainement.
- 2 fichiers .dat contenant les matrices RGB et Label pour la validation.

Extrants:
- Poids du modèle entrainé

Processus:
- Entrainement du modèle. 

## Fichier unet.py
Implantation du modèle [Unet](https://github.com/jocicmarko/ultrasound-nerve-segmentation). 

