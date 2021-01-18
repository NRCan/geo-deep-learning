# **Modules Documentation**

---

# Metrics in GDL

## Pixel based Metrics

#### Intersection over Union (IoU)

The IOU metric is a pixel based metric which measures overlap using the number of pixels common between groundtruth and predictions divided by the total pixels across both.

```
IoU=  (groundtruth ∩ prediction) /
      (groundtruth u predictions)
```

#### Dice Similarity Coefficient  (dice)

The dice metric scores model performance by measuring overlap between groundthruth and predictions divided by sum of pixels of both groundtruth and predictions.

```
dice= 2 * (groundtruth ∩ prediction) /
          (groundtruth + prediction)

```
_Note:_ IoU and Dice metrics weigh factors differently, however both metrics are positively correlated. This means if model A is better than B then this is captured similarly in both metrics.

#### Precision and Recall

By ploting a confusion matrix which indicates ground-truth and predicted classes with number of pixels classified in each class, [precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) and [recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) is easily computed.

```
precision= true positives /
           true positives + false positives
```

```
recall= true positives /
        true positives + false negatives
```

classes = A and B evaluating for class A
where,

- True Positives(TP) = pixels correctly classified as class A
- False Positives(FP) = pixels incorrectly classified as class A
- False Negatives(FN) = class A pixels incorrectly classified as class B
- True Negatives(TN) = pixels correctly classified as class B

#### Matthews Correlation Coefficient (MCC)

The MCC metric takes all four confusion matrix categories (TP, FP, FN and TN) into account which in turn provides a more reliable score. This [article](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7) compares MCC to other widely used metrics. This metric is insusceptible to the imbalanced dataset factor. Also [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) defines this metric is simpler terms.

```
MCC= TP * TN - FP * FN /
     [(TP + FP) * (FN + TN) * (FP + TN) * (TP + FN)]^(1/2)
```

#### Accuracy

Accuracy is simply defined here as the ratio of correctly classified pixels to total number of pixels between ground-truth and predicted.
_Note:_ This metric is very susceptible to imbalanced datasets and may give an overly optimistic score when that is not the case. We can write down the accuracy formular using the confusion matrix categories.

```
acc= TP + TN /
     TP + TN + FP + FN
```
## Comparing common pixel based metrics
| Accuracy   | IoU  | Dice Coefficient  |
|---|---|---|
| Counts the number of correctly classified pixels   | Counts pixels in both label and pred   | Similar to IoU, has its own strengths   |
| Not suitable for Imbalanced datasets    | Counts pixels in either label and pred  | Measures average performance to IoU’s measure of worst case performance.   |
| High accuracy may not translate to quality predictions   | Statistically correlates the counts (ratio)   |   |
|    | Accounts for imbalanced data by penalizing FP and FN   |   |

## Shape based Metrics

** New shape based metrics would be added soon **

---


## [Data_analysis Module](../utils/data_analysis.py)
The data_anlalysis module is used to visualize the composition of the sample's classes and see how it shapes the training dataset. Using basic statistical analysis, the user can test multiple sampling parameters and immediatly see their impact on the classes' distribution. It can also be used to automatically search optimal sampling parameters and obtain a more balanced class distribution in the dataset. The sampling parameters can then be used in images_to_samples.py to obtain the desired dataset. This way, there is no need to run images_to_samples.py to find out how the classes are distributed.

The data_analysis module is useful for balancing training data in which a class is under-represented.

### Prerequisites
Before running `data_analysis.py`, the paths to the csv file containing all the information about the images and the data folder must be specified in the `prep_csv_file` parameter and in the `data_path`parameter respectively. The `samples_size` and `samples_dist` must also be specified.

### data_analysis parameters in YAML file
Here is an example of the dedicated data_analysis section in the YAML file :

```YAML
# Data analysis parameters; used in data_analysis.py

data_analysis:
  create_csv: True
  optimal_parameters_search : False
  sampling: {'method':['min_annotated_percent', 'class_proportion'], 'map': 0, '0':0, '1':0, '2':0}
```
1. **create_csv**

      This parameter is used to create a csv file containing the class proportion data of each image sample. This first step is mandatory to ensure the proper operation of the module. Once it is created, the same csv file is used for every tests the user wants to perform. Once that is done, the parameter can then be changed to `False`.

      The `create_csv` parameter would have to be changed to `True` again if any changes were made to the content of the `prep_csv_file` or if the user wishes to change the values of the `samples_size` or `samples_dist` parameters. These parameters have a direct effect on the class proportion calculation.

      The csv file is stored in the folder specified in the `data_path` parameter of the YAML file.

1. **optimal_parameters_search**

    When this parameter is set to `True`, it activates the optimal sampling parameters search function. This function aims to find wich sampling parameters produce the most balanced dataset based on the standard deviation between the proportions of each class in the dataset.

    The sampling method(s) used for the search function must first be specified in the `sampling` dictionary in the `data_analysis` section of the YAML file. It does not take into account the values of the other keys in the dictionary.

    ###### Example
    ```YAML
    data_analysis:
      create_csv: False
      optimal_parameters_search : True
      sampling: {'method':['min_annotated_percent', 'class_proportion'], 'map': 0, '0':0, '1':0, '2':0}
    ```    
    <p align="center">
       <img align="center" src="/docs/screenshots/stats_parameters_search_map_cp.PNG">
    </p>

    The function first returns the optimal threshold(s) for the chosen sampling method(s). It then returns a preview of the proportions of each classes and the size of the final dataset without creating it.

1. **sampling dictionary**


    a) 'method'

    To specify the desired sampling method, write one or both of `'min_annotated_percent'` and `'class_proportion'`. These sampling methods can be used on their own or together in any order. They have to be in quotes and contained in a list.

    This part of the `sampling` dictionary is also used for the `optimal_parameters_search` function.


    b) 'map'

    'map' stands for 'minimum annotated percent' : Minimum percent of non background pixels in sample

    Specify the targeted mininum annotated percent with an integer. For this value to be taken into account, the `optimal_paramters_search` function must be turned off and 'min_annotated_percent' must be listed in the `'method'` key of the `sampling` dictionary.

     ###### Example

      ```YAML
        'map':25
      ```

    c) following keys

    The followings keys of the dictionary should be the number of each classes in the images in quotes. Specify the minimum class proportion of one or all classes with integer(s) or float(s). For these values to be taken into account, the `optimal_paramters_search` function must be turned off and 'class_proportion' must be listed in the `'method'` key of the `sampling` dictionary.

    ###### Example

      ```YAML
        '0':0, '1':10, '2':5, '3':5, ...
      ```

## Running data_analysis.py

To launch the program :

`python data_analysis.py path/to/config/file/config.yaml`
