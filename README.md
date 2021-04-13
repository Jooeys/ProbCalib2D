# ProbCalib2D
Sample × Category Probability Calibration in Two Dimensions Case

## Problem Setting
Deep Convolutional Neural Networks have achieved very good performance in text or image classification tasks. However, the problem of probability calibration leads to myriad problems in safety-critical machine learning application filed. The predictions of model can be over-confident if without calibration or mis-calibration.  As a consequence ,it is necessary to evaluate the model calibration [13]. There is still a main limitation, which is the calibration only adapted for one dimension. The aim is to find calibration methods that take into account both dimensions simultaneously. 

### Multi-label vs. Multi-class Classification: Sigmoid vs. Softmax
When designing a CNN model to perform a classification task (e.g. classifying objects in cifar10 dataset or classifying handwritten digits) we want to tell our model whether it is allowed to choose many answers (e.g. both frog and cat) or only one answer (e.g. the digit “8.”) 


This section will discuss how we can achieve this goal by applying either a sigmoid or a softmax function to our classifier’s raw output values.

### Applying Sigmoid versus Softmax
At the end of a neural network classifier, you’ll get a vector of “raw output values”: for example [-0.5, 1.2, -0.1, 2.4] if your neural network has four outputs . We’d like to convert these raw values into an understandable format: probabilities. After all, it makes more sense to tell a patient that their risk of diabetes is 91\% rather than “2.4” (which looks arbitrary.)

We convert a classifier’s raw output values into probabilities using either a sigmoid function or a softmax function.


### Applying SoftMax normalization versus Platt Scaling} These scores can be used for ranking test images according to their likeliness to contain a given target concept (search task) or for ranking target concepts according to their likeliness to be visible in a given image (classification task). 

## Evaluating Calibration 
In order to rectify the problem of the uncertainty of model, these works resulted in two common ways of measuring calibration: reliability diagrams [17] and estimates of the squared expected calibration error (ECE)[17].

Both tasks are usually evaluated with different metrics. In the multi-class setting, the classification performance is generally evaluated using the top-N accuracy, usually with N = 1.

The retrieval performance is generally evaluated using the mean average precision (MAP).
The MAP may also be used for the evaluation of the classification performance in the multilabel setting, like for Pascal VOC, as there may be more than one correct category associated to a given sample
