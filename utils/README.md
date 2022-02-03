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
<!--
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
-->
## Comparing common pixel based metrics
| Accuracy   | IoU  | Dice Coefficient  |
|---|---|---|
| Counts the number of correctly classified pixels   | Counts pixels in both label and pred   | Similar to IoU, has its own strengths   |
| Not suitable for Imbalanced datasets    | Counts pixels in either label and pred  | Measures average performance to IoU’s measure of worst case performance.   |
| High accuracy may not translate to quality predictions   | Statistically correlates the counts (ratio)   |   |
|    | Accounts for imbalanced data by penalizing FP and FN   |   |

## Shape based Metrics

** New shape based metrics would be added soon ** 