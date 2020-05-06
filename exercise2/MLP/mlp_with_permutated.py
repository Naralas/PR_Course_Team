import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from warnings import filterwarnings

from datasets import paths, read_png

import statistics
from collections import defaultdict


def find_best_model():
    print("Importing train set...")
    x_train, y_train = read_png(paths["png-train"])
    print("Train set size:", len(x_train))

    parameters = {
        "activation": ["logistic"],
        "learning_rate": ["constant"],
        "solver": ["sgd"],
        "hidden_layer_sizes": [int(i) for i in np.linspace(20, 100, 4)],
        "max_iter": [int(i) for i in np.linspace(100, 1000, 5)],
        "learning_rate_init": [float(i) for i in np.logspace(-3, -1, 3)]
    }

    clf = GridSearchCV(MLPClassifier(), parameters,
                       return_train_score=True, cv=4, n_jobs=8)

    print("Training...")
    clf.fit(x_train, y_train)

    results = clf.cv_results_
    train_score = get_metric_by_iter(results, "mean_train_score")
    test_score = get_metric_by_iter(results, "mean_test_score")

    plt.figure()
    plt.title("Accuracy over epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")

    epochs, train_max, train_mean, train_std = get_stats_for_metrics(
        train_score)
    plt.errorbar(epochs, train_mean, yerr=train_std,
                 label="Train: average/stdev")
    plt.plot(epochs, train_max, label="Train: max")

    epochs, test_max, test_mean, test_std = get_stats_for_metrics(test_score)
    plt.errorbar(epochs, test_mean, yerr=test_std, label="Test: average/stdev")
    plt.plot(epochs, test_max, label="Test: max")

    plt.legend()
    plt.savefig("img/b/accuracy_over_epochs.png")

    print(f"Best parameters: {clf.best_params_}")
    print(f"Best score: {clf.best_score_}")

    return clf.best_estimator_


def get_metric_by_iter(results, metric_name):
    metric_by_iter = defaultdict(list)
    for i, value in enumerate(results["param_max_iter"]):
        metric = results[metric_name][i]
        metric_by_iter[value].append(metric)
    metric_by_iter = dict(metric_by_iter)
    return metric_by_iter


def get_stats_for_metrics(metric_by_iter):
    epochs = list(metric_by_iter.keys())
    max_by_epochs = [max(
        v) for k, v in metric_by_iter.items()]
    mean_by_epochs = [statistics.mean(
        v) for k, v in metric_by_iter.items()]
    std_by_epochs = [statistics.stdev(
        v) for k, v in metric_by_iter.items()]
    return epochs, max_by_epochs, mean_by_epochs, std_by_epochs


def evaluate(model):
    print("Evaluating model on standard dataset")
    print("Importing train set...")
    x_train, y_train = read_png(paths["png-train"])
    print("Train set size:", len(x_train))
    print("Training...")
    model.fit(x_train, y_train)

    print("Importing validation set...")
    x_val, y_val = read_png(paths["png-val"])
    print("Validation set size:", len(x_val))

    y_pred_val = model.predict(x_val)

    accuracy = accuracy_score(y_val, y_pred_val)
    print("Accuracy on validation set :", accuracy)


def evaluate_permutated(model):
    print("Evaluating model on permutated dataset")
    print("Importing train set...")
    x_train, y_train = read_png(paths["png-perm-train"])
    print("Train set size:", len(x_train))
    print("Training...")
    model.fit(x_train, y_train)

    print("Importing validation set...")
    x_val, y_val = read_png(paths["png-perm-val"])
    print("Validation set size:", len(x_val))

    y_pred_val = model.predict(x_val)

    accuracy = accuracy_score(y_val, y_pred_val)
    print("Accuracy on validation set :", accuracy)


def get_best_model():
    params = {
        "activation": "logistic",
        "hidden_layer_sizes": 100,
        "learning_rate": "constant",
        "learning_rate_init": 0.1,
        "max_iter": 550,
        "solver": "sgd",
    }
    # Best score: 0.971592039800995
    model = MLPClassifier(**params)
    return model


if __name__ == "__main__":
    filterwarnings("ignore")
    # model = find_best_model()
    model = get_best_model()
    evaluate(model)
    print()
    evaluate_permutated(model)
