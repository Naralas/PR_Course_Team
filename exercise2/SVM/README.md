# Task 2a

We provide a completed run Jupyter notebook in HTML format at the root of the `/SVM/` folder.
## Setup and execution

You need to download the MNIST dataset as provided on [Ilias](https://ilias.unibe.ch/goto_ilias3_unibe_fold_1760165.html) and extract the csv files to the `/SVM/data/` folder.

Then you need to install the required python modules to run the project. We suggest doing it in a virtualenv but you probably know that already.

For example at the root of the `SVM/` folder : 
- `virtualenv venv`
- `source venv/Scripts/activate`
- `pip install -r requirements.txt`

You can then start the jupyter notebook using `jupyter notebook`, select the `svm.ipynb` notebook and run the cells.


## Results and observations

Using grid-search, we have found that the best kernel to use was the polynomial one, with degree 2. This resulted in an accuracy on the test set of **98.06 %**.

We wanted to try more hyper-parameters, but unfortunately, the computing time using the `GridSearchCV` of `sklearn` with **cross-validation** already took more than an hour and half. 

A possible solution would be to reduce the size of the dataset, making sure the classes are balanced and computing on the subset.