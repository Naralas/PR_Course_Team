import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import cross_val_score

# SVM

#Loading csv data to numpy arrays
def loadData(file, label=True):
    csv = open(file,"r")
    array = []
    for line in csv:
        array.append(line.strip().split(","))
    array = np.asarray(array, dtype=np.int)
    if(label):
        samples =   array[:,1:]
        labels = array[:,0]
        return labels, samples
    else:
        samples = array[:,:]
        return samples

#Loading Training Set
def loadTrainSet():
    return loadData("train.csv")

#Loading Test Set
def loadTestSet():
    # return loadData("Validation/mnist_test.csv")
    return loadData("test.csv")

def loadValidationTestSet():
    return loadData("Validation/mnist_test.csv", False)

#Application
cross_validation = False # True or False.

print ("Application started. Loading data...")
start = int(round(time.time() * 1000))
y_train, X_train = loadTrainSet()
print ("Training set loaded with data size: ", len(X_train))
if(not cross_validation):
    # y_test, X_test = loadTestSet()
    X_test = loadValidationTestSet()
    print ("Test set loaded with data size: ", len(X_test))
print ("Data loaded in ",int(round(time.time() * 1000)) - start," ms. ")

print ("Starting with model generation.")
calcStart = int(round(time.time() * 1000))
#clf = svm.SVC(kernel='linear', C = 0.0001)
clf = svm.SVC(kernel='poly', gamma=1.0, C = 1.0, degree=2)
#clf = svm.SVC(kernel='rbf', gamma=0.001, C = 1.0)
#clf = svm.SVC(kernel='sigmoid', gamma=0.0000001, C = 1.0)
predictions = []
if(not cross_validation):
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print ("Model generated. Time taken for calculations: ",int(round(time.time() * 1000)) - calcStart, " ms")
    #print("Accuracy: ", clf.score(X_test,y_test))
else:
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print("Average Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

print ("Application finished in : ",int(round(time.time() * 1000)) - start, " ms")

validation_file_name = "predictions.txt"
validation_file = open("Validation/" + validation_file_name, "w")
for pred_id, prediction in enumerate(predictions):
    validation_file.write(str(pred_id) + ", " + str(prediction) + "\n")
    # print(str(pred_id) + ", " + str(prediction))
validation_file.close()

unique, counts = np.unique(predictions, return_counts=True)
summary = dict(zip(unique, counts))
print(summary);