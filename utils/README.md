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
