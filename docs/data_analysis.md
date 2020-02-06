# **Data_analysis Module**
The data_anlalysis module is used to visualize the distribution of the sample's classes and see how it shapes the training dataset. Using basic statistical analysis, the user can test multiple sampling parameters and immediatly see their impact on the classes' distribution. It can also be used to automatically search optimal sampling parameters and obtain a more balanced class distribution in the dataset. The sampling parameters can then be used in images_to_samples.py to obtain the desired dataset. This way, there is no need to run images_to_samples.py to find out how the classes are distributed.

The data_analysis module is useful for balancing training data in which a class is under-represented. 

## Using data_analysis.py

### Prerequisites
Before running `data_analysis.py`, the paths to the csv file containing all the information about the images and the data folder must be specified in the `prep_csv_file` parameter and in the `data_path`parameter respectively. The `samples_size` and `samples_dist` must also be specified.

### data_analysis parameters in YAML file
Here is an example of the dedicated data_analysis section in the YAML file :

```YAML
# Data analysis parameters; used in data_analysis.py

data_analysis:
  create_csv: True
  optimal_parameters_search : True
  sampling: {'method':['min_annotated_percent', 'class_proportion'], 'map': 0, '0':0, '1':0, '2':0}
```
1. **create_csv**

      This parameter is used to create a csv file containing the class proportion data of each image sample. This first step is mandatory to ensure the proper operation of the module. Once it is created, the same csv file is used for every tests the user wants to perform. The parameter can then be changed to `False`.
      
      The `create_csv` parameter would have to be changed to `True` again if any changes were made to the content of the `prep_csv_file` or if the user wishes to change the values of the `samples_size` or `samples_dist` parameters. These parameters have a direct effect on the class proportion calculation.
      
      The csv file is stored in the folder specified in the `data_path` parameter of the YAML file.
      
1. **optimal_parameters_search**

    When this parameter is set to `True`, it activates the optimal sampling parameters search function. This function aims to find wich sampling parameters produce the most balanced dataset based on the standard deviation between the proportions of each class in the dataset. This function returns the best sampling parameter(s) value(s) depending on the sampling method(s) chosen.

1. **sampling dictionnary**
    1. 'method'
    1. 'map'
    1. following keys
    
    
