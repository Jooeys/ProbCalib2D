
# probability-calibration Python Library
Sample × Category Probability Calibration with Two Dimensions (ProbCalib2D)

This repository contains library code (Calibration Folder) to evaluate calibration and to measure the calibration error of models, including confidence intervals, models outcome probability give more information, not traditionally measure on model's accuracy. Generally, calibration is a pos-processing way to take an existing model and correct its uncertainties to make them more reliable and trustworthy.

## Problem Setting

Deep Convolutional Neural Networks have achieved very good performance in text or image classification tasks. However, the problem of probability calibration leads to myriad problems in safety-critical machine learning application filed. The predictions of model can be over-confident if without calibration or mis-calibration.  As a consequence ,it is necessary to evaluate the model calibration. There is still a main limitation, which is the calibration only adapted for one dimension. The aim is to find calibration methods that take into account both dimensions simultaneously. 

### Installation
```
pip install probability-calibration
```
https://pypi.org/project/probability-calibration/

### Usage(examples)
```
import Calibration
Calibration.output()
```
### Multi-label vs. Multi-class Classification
**Multi-label**: In the complete and “non-exclusive”,  or multi-label setting,  zero,one or more categories can be associated to a given sample (e.g., Pascal VOC [3]). The labelvector associated to a given sample is still a binary one but does not necessarily correspond toany one-hot encoding

Multi-label classification is a predictive modeling task that involves predicting zero or moremutually non-exclusive class labels. When designing a CNN model to perform a classificationtask (e.g. classifying objects in cifar10 dataset or classifying handwritten digits) we want to tellour model whether it is allowed to choose many answers (e.g.  both frog and cat) or only oneanswer (e.g. the digit “8.”

**Multi-class**: In the complete and “exclusive”, or multi-class setting, each of thesamples is associated to exactly one category (e.g., MNIST [2], SVHN [5], CIFAR [4], CUBcategories [8],ILSVRC [7], and many other standard collections).
This section will discuss how we can achieve this goal by applying either a sigmoid or a softmax function to our classifier’s raw output values.

### Applying Sigmoid versus Softmax
Generally, we use softmax activation instead of sigmoid with the cross-entropy loss because softmax activation distributes the probability throughout each output node. If it is a binary classification, using sigmoid is same as softmax. For multi-class classification use sofmax with cross-entropy.

At the end of a neural network classifier, you’ll get a vector of “raw output values”: for example [-0.5, 1.2, -0.1, 2.4] if your neural network has four outputs . We’d like to convert these raw values into an understandable format: probabilities. After all, it makes more sense to tell a patient that their risk of diabetes is 91\% rather than “2.4” (which looks arbitrary.)

We convert a classifier’s raw output values into probabilities using either a sigmoid function or a softmax function.


### Applying SoftMax normalization versus Platt Scaling
Is Platt scaling can well calibrated? For anwsering this question, we are going to take Softmax to compare methods like Platt scaling. These scores can be used for ranking test images according to their likeliness to contain a given target concept (search task) or for ranking target concepts according to their likeliness to be visible in a given image (classification task). 

## Evaluating Calibration 
In order to rectify the problem of the uncertainty of model, these works resulted in two common ways of measuring calibration: reliability diagrams [17] and estimates of the squared expected calibration error (ECE)[17].

Both tasks are usually evaluated with different metrics. In the multi-class setting, the classification performance is generally evaluated using the top-N accuracy, usually with N = 1.

The retrieval performance is generally evaluated using the mean average precision (MAP).
The MAP may also be used for the evaluation of the classification performance in the multilabel setting, like for Pascal VOC, as there may be more than one correct category associated to a given sample

Visualizing calibration with reliability diagrams.