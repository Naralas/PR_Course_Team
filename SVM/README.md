# Task 2a

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
