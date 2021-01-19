# **Modules Documentation**

---

## **[Metrics](metrics.py)**
Metrics are used to measure the quality of the statistical or the deep learning model. Evaluating deep learning models or algorithms is essential for any project. There some metrics available in **GDL**.

### Pixel based Metrics

#### Intersection over Union (IoU)

The IOU metric is a pixel based metric which measures overlap using the number of pixels common between the ground truth and the predictions divided by the total pixels across both.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\textrm{IoU}=\frac{\textrm{ground truth} \cap \textrm{prediction}}{\textrm{ground truth} \cup \textrm{prediction}}" title="\Large IoU" class="center" />
</p>

#### Dice Similarity Coefficient  (dice)

The dice metric scores model performance by measuring overlap between the ground truth and predictions divided by sum of pixels of both ground truth and predictions.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\textrm{Dice}=2\cdot \frac{\textrm{ground truth} \cap \textrm{prediction}}{\textrm{ground truth} + \textrm{prediction}}" title="\Large Dice" class="center" />
</p>

>**Note :** IoU and Dice metrics weigh factors differently, however both metrics are positively correlated. This means if model A is better than B then this is captured similarly in both metrics.

#### Precision and Recall

By plotting a confusion matrix which indicates ground-truth and predicted classes with number of pixels classified in each class, [precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) and [recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) is easily computed.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\textrm{precision}=\frac{\textrm{true positives} }{\textrm{true positives} + \textrm{false positives}}" title="\Large Precision" class="center" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\textrm{recall}=\frac{\textrm{true positives}}{\textrm{true positives} + \textrm{false negatives}}" title="\Large Recall" class="center" />
</p>

<!-- classes = A and B evaluating for class A -->
Where,
- True Positives (TP) = pixels correctly classified as class A
- False Positives (FP) = pixels incorrectly classified as class A
- False Negatives (FN) = class A pixels incorrectly classified as class B
- True Negatives (TN) = pixels correctly classified as class B

#### Matthews Correlation Coefficient (MCC)

The MCC metric takes all four confusion matrix categories (TP, FP, FN and TN) into account which in turn provides a more reliable score. This [article](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7) compares MCC to other widely used metrics. This metric is insusceptible to the imbalanced dataset factor. Also [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) defines this metric is simpler terms.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\textrm{MCC}=\frac{(\textrm{TP}\cdot \textrm{TN}) - (\textrm{FP}\cdot \textrm{FN})}{\sqrt{(\textrm{TP}+\textrm{FP})(\textrm{TP}+\textrm{FN})(\textrm{TN}+\textrm{FP})(\textrm{TN}+\textrm{FN})}}" title="\Large MCC" class="center" />
</p>

#### Accuracy
Accuracy is simply defined here as the ratio of correctly classified pixels to total number of pixels between ground-truth and predicted.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\textrm{acc}=\frac{\textrm{TP}+\textrm{TN}}{\textrm{TP}+\textrm{TN}+\textrm{FP}+\textrm{FN}}" title="\Large Acc" class="center" />
</p>

>**Note :** This metric is very susceptible to imbalanced datasets and may give an overly optimistic score when that is not the case. We can write down the accuracy formular using the confusion matrix categories.


#### Comparing common pixel based metrics
| Accuracy   | IoU  | Dice Coefficient  |
|:----------:|:----:|:-----------------:|
| Counts the number of correctly classified pixels   | Counts pixels in both label and pred   | Similar to IoU, has its own strengths   |
| Not suitable for Imbalanced datasets    | Accounts for imbalanced data by penalizing FP and FN  | Measures average performance to IoU’s measure of worst case performance.   |
| High accuracy may not translate to quality predictions   | Statistically correlates the counts (ratio)   |   |


### Shape based Metrics

**New shape based metrics would be added soon**

---

## **[Optimizers](optimizer.py)**
For training a neural network to minimize the losses so as to perform better, we need to tweak the weights and parameters associated with the model and the loss function. This is where optimizers play a crucial role. Here some optimizers available in **GDL**.


TODO


---

## **[Data Analysis Module](data_analysis.py)**
The [data_analysis](data_analysis.py) module is used to visualize the composition of the sample's classes and see how it shapes the training dataset. Using basic statistical analysis, the user can test multiple sampling parameters and immediately see their impact on the classes' distribution. It can also be used to automatically search optimal sampling parameters and obtain a more balanced class distribution in the dataset. The sampling parameters can then be used in [images_to_samples.py](../images_to_samples.py) to obtain the desired dataset. This way, there is no need to run [images_to_samples.py](../images_to_samples.py) to find out how the classes are distributed.

The [data_analysis](data_analysis.py) module is useful for balancing training data in which a class is under-represented.

### Prerequisites
Before running [data_analysis.py](data_analysis.py), the paths to the `csv` file containing all the information about the images and the data folder must be specified in the `yaml` file use to the experience.
```YAML
# Global parameters
global:
  samples_size: 512
  num_classes: 5
  data_path: path/to/data                 # <--- must be specified
  number_of_bands: 4

      ...

# Sample parameters; used in images_to_samples.py -------------------
sample:
  prep_csv_file: /path/to/csv/images.csv  # <--- must be specified
  val_percent: 5

```


### Data Analysis parameters in the `YAML` file
Here is an example of the dedicated data_analysis section in the YAML file :

```YAML
# Data analysis parameters; used in data_analysis.py
data_analysis:
  create_csv: True
  optimal_parameters_search : False
  sampling_method: # class_proportion or min_annotated_percent
    'min_annotated_percent': 0  # Min % of non background pixels in samples.
    'class_proportion': {'1':0, '2':0, '3':0, '4':0} # See below:
    # keys (numerical values in 'string' format) represent class id
    # values (int) represent class minimum threshold targeted in samples
```
#### Parameters
**`create_csv` :**
This parameter is used to create a `csv` file containing the class proportion data of each image sample. This first step is mandatory to ensure the proper operation of the module. Once it is created, the same `csv` file is used for every tests the user wants to perform. Once that is done, the parameter can then be changed to `False`.
This parameter would have to be changed to `True` again if any changes were made to the content of the `prep_csv_file` or if the user wishes to change the values of the `samples_size` parameters. These parameters have a direct effect on the class proportion calculation. The `csv` file created is stored in the folder specified in the `data_path` of the global section.

**`optimal_parameters_search` :**
When this parameter is set to `True`, it activates the optimal sampling parameters search function. This function aims to find which sampling parameters produce the most balanced dataset based on the standard deviation between the proportions of each class in the dataset. The sampling method(s) used for the search function must first be specified in the `sampling_method` dictionary in the `data_analysis` section. It does not take into account the values of the other keys in the dictionary. The function first returns the optimal threshold(s) for the chosen sampling method(s). It then returns a preview of the proportions of each classes and the size of the final dataset without creating it, like the following image.
<p align="center">
   <img align="center" src="/docs/screenshots/stats_parameters_search_map_cp.PNG">
</p>

**`sampling_method` :**
For `min_annotated_percent` is the minimum percent of non background pixels in the samples. By default the value is `0` and the targeted minimum annotated percent must by a integer.
`class_proportion` should be a dictionary with the number of each classes in the images in quotes. Specify the minimum class proportion of one or all classes with integer(s) or float(s). Example, `'0':0, '1':10, '2':5, '3':5, ...`
<!-- `min_annotated_percent`, For this value to be taken into account, the `optimal_paramters_search` function must be turned off and 'min_annotated_percent' must be listed in the `'method'` key of the `sampling` dictionary. -->
<!-- `class_proportion`, For these values to be taken into account, the `optimal_paramters_search` function must be turned off and 'class_proportion' must be listed in the `'method'` key of the `sampling` dictionary. -->

#### Running [data_analysis.py](data_analysis.py)
You can run the data analysis alone if you only want the stats, and to do that you only need to launch the program :
```shell
python data_analysis.py path/to/yaml_files/your_config.yaml
```
