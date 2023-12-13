#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:00:39 2023

@author: amaterasu
"""

import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

digits = datasets.load_digits()
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data)) 
rng.shuffle(indices)

X = digits.data[indices[:330]]
y = digits.target[indices[:330]] 
images = digits.images[indices[:330]]

n_total_samples = len(y)
n_labeled_points = 10
unlabeled_set = np.arange(n_total_samples)[n_labeled_points:]

class LabelPropagationModel:
    def __init__(self, X, y, images, unlabeled_set, n_labeled_points):
        self.X = X
        self.y = y
        self.images = images
        self.unlabeled_set = unlabeled_set
        self.n_labeled_points = n_labeled_points

    def label_propagation(self, tolerance=0.001, max_iter=300):
        graph_matrix = kneighbors_graph(self.X, 7, mode='connectivity', include_self=True)

        classes = np.unique(self.y)
        classes = classes[classes != -1]
        
        y_train = np.copy(self.y)
        y_train[self.unlabeled_set] = -1

        unlabeled = y_train == -1

        Y1 = np.zeros((len(y_train), len(classes)))
        for label in classes:
            Y1[y_train == label, classes == label] = 1

        Y0 = np.copy(Y1)
        Y_prev = np.zeros((self.X.shape[0], len(classes)))

        unlabeled = unlabeled[:, np.newaxis]

        for n_iter_ in range(max_iter):
            if np.abs(Y1 - Y_prev).sum() < tolerance:
                break

            Y_prev = Y1
            Y1 = graph_matrix @ Y1
            normalizer = np.sum(Y1, axis=1)[:, np.newaxis]
            normalizer[normalizer == 0] = 1
            Y1 /= normalizer

            Y1 = np.where(unlabeled, Y1, Y0)

        F_u = classes[np.argmax(Y1, axis=1)]
        return F_u, classes, Y1

    def print_model(self, true_labels, predicted_labels, labels):
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        accuracy = accuracy_score(true_labels, predicted_labels)

        print(
            "Label propagation model: %d labeled & %d unlabeled points (%d total)"
            % (self.n_labeled_points, len(self.y) - self.n_labeled_points, len(self.y))
        )
        print("Accuracy: ",'{:.1%}'.format(accuracy))
        print("Confusion matrix")
        print(cm)
        print("\n")

        self.plot_confusion_matrix(cm, labels)

    def plot_confusion_matrix(self, cm, labels):
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.show()



    def run(self, iterations=5, high_confidence=True):
        for i in range(iterations):
            y_train = np.copy(self.y)
            y_train[self.unlabeled_set] = -1

            F_u, labels, Y1 = self.label_propagation()

            predicted_labels = F_u[self.unlabeled_set]
            true_labels = self.y[self.unlabeled_set]

            self.print_model(true_labels, predicted_labels, labels)

            if high_confidence:
                pred_entropies = stats.distributions.entropy(Y1.T)
                certain_index = np.argsort(pred_entropies)
            else:
                pred_entropies = stats.distributions.entropy(Y1.T)
                certain_index = np.argsort(pred_entropies)[::-1]

            certain_index = certain_index[np.in1d(certain_index, self.unlabeled_set)][:5]


            self.unlabeled_set = np.setdiff1d(self.unlabeled_set, certain_index)
            self.n_labeled_points += len(certain_index)
            
            
            
model = LabelPropagationModel(X, y, images, unlabeled_set, n_labeled_points)

# Run the model for high confidence predictions
#model.run(iterations=5, high_confidence=True)

# Run the model for low confidence predictions
model.run(iterations=5, high_confidence=False)



