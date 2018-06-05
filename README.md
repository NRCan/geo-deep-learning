# CCMEO's deep learning project

## Requirements  
- Python 3.6 with the following libraries:
    - pytorch 0.4.0
    - torchvision 0.4.0
    - numpy 1.14.0
    - gdal 2.2.2
    - ruamel_yaml 0.15.35
    - scikit-image 0.13.1
    - PIL 5.1.0
- nvidia GPU highly recommended

## images_to_samples.py  
To launch the program:  
``` 
python images_to_samples.py path/to/parameter_file.txt
```  
Parameter file content:  
```
path/to/folder/images_and_references
path/to/folder/output_samples
size of each sample (in pixel)
distance between 2 sample centers
```

Outputs:
- 2 .dat files with RGB and reference arrays
- 1 .txt file with number of samples created and the number of classes in the references data.

Process: 
- Read images in the "RGB" and the "label" folders
- Convert images to arrays
- Divide images in samples of size and distance specified in the parameters
- Write samples in 2 .dat files (RGB and label)

## train_model.py
To launch the program:  
``` 
python train_model.py path/to/parameter_file.txt
```  
Parameter file content:  
```
path/to/folder/training_and_validation_data
batch size
number of epoch
learning rate
size of each sample (in pixel)
number of classes
number of training samples
number of validation samples
```

Inputs:
- 2 .dat files with RGB and reference arrays use for training
- 2 .dat files with RGB and reference arrays use for validation

Output:
- Trained model weights

Process:
- model training. 


