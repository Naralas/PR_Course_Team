# PR_Course_Team
Repository for the team projects in the pattern recognition course, MCS 2020 UniFr.<br>
Team "chaussette".
## Task 2a

We provide a completed run Jupyter notebook in HTML format at the root of the `/SVM/` folder.
### Setup and execution

You need to download the MNIST dataset as provided on [Ilias](https://ilias.unibe.ch/goto_ilias3_unibe_fold_1760165.html) and extract the csv files to the `/SVM/data/` folder.

Then you need to install the required python modules to run the project. We suggest doing it in a virtualenv but you probably know that already.

For example at the root of the `SVM/` folder : 
- `virtualenv venv`
- `source venv/Scripts/activate`
- `pip install -r requirements.txt`

You can then start the jupyter notebook using `jupyter notebook`, select the `svm.ipynb` notebook and run the cells.


### Results and observations

Using grid-search, we have found that the best kernel to use was the polynomial one, with degree 2. This resulted in an accuracy on the test set of **98.06 %**.

We wanted to try more hyper-parameters, but unfortunately, the computing time using the `GridSearchCV` of `sklearn` with **cross-validation** already took more than an hour and half. 

A possible solution would be to reduce the size of the dataset, making sure the classes are balanced and computing on the subset.

## Task 2b - Multilayer perceptron

### Overview

The model is trained and tested with a lot of different values (125) for the following parameters:
- number of neurons (values = 20,40,60,80,100)
- learning rate (values = 0.02, 0.04, 0.06, 0.08, 0.1)
- training epochs (values = 200, 400, 600, 800, 1000).

### Best results

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


## Task 2c - Convolutional Neural Network

### Plots
![Plots](./plots.png)

### Best achieved accuracy
The maximum accuracy was achieved with lr = 0.001, on epoch 7 with 98.36% accuracy
