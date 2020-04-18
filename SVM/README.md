# Task 2a

We provide a completed run Jupyter notebook in HTML format at the root of the `/SVM/` folder.
## Setup

You need to download the MNIST dataset as provided on [Ilias](https://ilias.unibe.ch/goto_ilias3_unibe_fold_1760165.html) and extract the csv files to the `/SVM/data/` folder.

Then you need to install the required python modules to run the project. We suggest doing it in a virtualenv but you probably know that already.

For example at the root of the `SVM/` folder : 
- `virtualenv venv`
- `source venv/Scripts/activate`
- `pip install -r requirements.txt`

## How to run

You can then start the jupyter notebook using `jupyter notebook`, select the `svm.ipynb` notebook and run the cells.

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
