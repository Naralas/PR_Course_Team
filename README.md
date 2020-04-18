# PR_Course_Team
Repository for the team projects in the pattern recognition course, MCS 2020 UniFr.

# Task 2a (SVM)

## How to run

You can open source file to modify parameters cross_validation and svm kernels. If cross validation is enabled,
then it will perform cross validation on training data. Otherwise, it will run on test set.

    svm.py

## Results
You can see the output on svm.ipynb.


## Conclusions

As C value increases the execution time of the program increases as well.

Highest accuracy we can get with sigmoid is around 0.9102 and rbf is around 0.9649.

In comparison of execution times, sigmoid and rbf spends similar time. Polynomial kernel
performs nearly same with linear kernel and they are faster than sigmoid and rbf.

# Task 2b - Multilayer perceptron

## Overview

The model is trained and tested with a lot of different values (125) for the following parameters:
- number of neurons (values = 20,40,60,80,100)
- learning rate (values = 0.02, 0.04, 0.06, 0.08, 0.1)
- training epochs (values = 200, 400, 600, 800, 1000).

## Best results

Best accuracy (92.08%) obtained with the following parameters:
- number of neurons = 100
- learning rate = 0.02
- training epochs = 800

**The following diagrams show the accuracy rate of the model with the best found parameters on the traing and validation set with respect to the training epochs**

**Green plot is the accuracy on the training set and the orange one is on the validation set.**

First iteration
![](./1_fold.PNG)

Second iteration
![](./2_fold.PNG)

Third iteration
![](./3_fold.PNG)

Fourth iteration
![](./4_fold.PNG)


# Task 2c - Convolutional Neural Network

## Plots
![Plots](./plots.png)

## Best achieved accuracy
The maximum accuracy was achieved with lr = 0.001, on epoch 7 with 98.36% accuracy
