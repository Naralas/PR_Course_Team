import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def read_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    matrix = np.array(data, dtype=int)
    samples = matrix[:, 1:]
    labels = matrix[:, 0]
    return labels, samples


def main():
    train_filename = 'mnist_train.csv'
    test_filename = 'mnist_test.csv'

    train_labels, train_samples = read_data(train_filename)
    best_classifier = None
    max_accuracy = 0
    for iter in range(5):
        iteration = (iter + 1) * 200
        for neu in range(5):
            neurons = (neu + 1) * 20
            for lr in range(5):
                learning_rate = (lr + 1) * 0.02
                classifier = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(neurons,),
                                           max_iter=iteration, learning_rate='constant',
                                           learning_rate_init=learning_rate)
                k_folds = KFold(n_splits=4)
                scores = []
                for train, test in k_folds.split(train_samples):
                    train_samples_cv = train_samples[train]
                    train_labels_cv = train_labels[train]

                    """Train the MLP"""
                    trained_mlp = classifier.fit(train_samples_cv, train_labels_cv)

                    """Test the classifier"""
                    test_samples_cv = train_samples[test]
                    test_labels_cs = train_labels[test]
                    score = trained_mlp.score(test_samples_cv, test_labels_cs)
                    scores.append(score)
                print('For iterations = {iter}'.format(iter=iteration))
                print('For number of neurons on the hidden layer = {neurons}'.format(neurons=neurons))
                print('For learning rate = {learning_rate}'.format(learning_rate=learning_rate))
                print('The following accuracies are obtained for four-fold CV: {acc}'.format(acc=scores))
                accuracy = np.mean(scores)
                print('Average accuracy = {acc:.4f}'.format(acc=accuracy))
                print('---------------------------------------------------')
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_classifier = classifier
    print('Best accuracy found with the MLP with the following parameters: {params}'.format(params=best_classifier))

    """Get the accuracy on the testset with the best parameters found during the training phase of the MLP"""
    test_labels, test_samples = read_data(test_filename)
    best_classifier.fit(train_samples, train_labels)
    score = best_classifier.score(test_samples, test_labels)
    print('Accuracy = {acc}'.format(acc=score))

    """Plot the accuracy on train & validation dataset for the best classifier."""
    neurons = best_classifier.hidden_layer_sizes[0]
    learning_rate = best_classifier.learning_rate_init

    k_folds = KFold(n_splits=4)

    for train, validation in k_folds.split(train_samples):
        accuracies_training_dataset = []
        accuracies_validation_dataset = []
        epoches = []

        training_samples = train_samples[train]
        training_labels = train_labels[train]

        test_samples = train_samples[validation]
        test_labels = train_labels[validation]

        for iter in range(5):
            iteration = (iter + 1) * 200

            best_classifier = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(neurons,),
                                            max_iter=iteration, learning_rate='constant',
                                            learning_rate_init=learning_rate)

            best_classifier.fit(training_samples, training_labels)

            score = best_classifier.score(test_samples, test_labels)
            epoches.append(iteration)
            accuracies_training_dataset.append(score)

            score = best_classifier.score(training_samples, training_labels)
            accuracies_validation_dataset.append(score)

        plt.plot(epoches, accuracies_training_dataset, color='green')
        plt.plot(epoches, accuracies_validation_dataset, color='orange')
        plt.xlabel('Epoche')
        plt.ylabel('Accuracy')
        plt.show()


if __name__ == '__main__':
    main()
