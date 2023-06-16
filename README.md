# ERA1_Session7
 Session 7 Assignment is to build a neural net model with less than 8000 parameters that can achieve accuracy of 99.4% on MNIST digit dataset within 15 training epochs. In the following 3 steps, we will build a model that will achive this target.

 ## STEP 1 (Notebook [session7_1](https://github.com/sdev2030/ERA1_Session7/blob/main/session7_1.ipynb))
 
**Target:**

Setup a skeletal model (Copied from 3rd example from the class)

**Results:**

Parameters: 10.7k

Best Train Accuracy: 98.56

Best Test Accuracy: 98.51

**Analysis:**

Model with all the required functions

No over-fitting, model is capable of improvement

## STEP 2 (Notebook [session7_2](https://github.com/sdev2030/ERA1_Session7/blob/main/session7_2.ipynb))
**Target:**

Add Regularization, BatchNorm and dropout to improve model efficiency
Add global average pooling and remove big 7x7 kernel, adjust model parameters to be close to 7k

**Results:**

Parameters: 7k

Best Train Accuracy: 98.57

Best Test Accuracy: 99.07

**Analysis:**
Accuracy has not suffered in spite of reducing parameters with GAP

Could increase model capacity

No over-fitting, model is capable of improvement
## STEP 3 (Notebook [session7_3](https://github.com/sdev2030/ERA1_Session7/blob/main/session7_3.ipynb))
**Target:**

Add data augmentation and LR scheduler to get better model

Increase model size to be close to 8k

**Results:**

Parameters: 7.9k

Best Train Accuracy: 99.30

Best Test Accuracy: 99.53

**Analysis:**

Slight under-fitted model, is capable if pushed further with additional training and data augmentation techniques.