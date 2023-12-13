#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:47:38 2023

@author: amaterasu
"""


import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph

n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False)
outer , inner = 0 , 1
labels = np.full(n_samples, -1.0)
labels[0] = outer
labels[-1] = inner


class GraphLabelPropagation:
    def __init__(self, X, labels, outer, inner):
        self.X = X
        self.labels = labels
        self.outer = outer
        self.inner = inner

    def plotScatter(self, labels, title="", outerTitle="", innerTitle="", legend=True):
        plt.figure(figsize=(6, 6))

        plt.scatter(
            self.X[labels == self.outer, 0],
            self.X[labels == self.outer, 1],
            color="c",      
            marker="*",
            label=outerTitle,    
        )
        plt.scatter(
            self.X[labels == self.inner, 0],
            self.X[labels == self.inner, 1],
            color="r",    
            marker="*",  
            label=innerTitle,    
        )
        plt.scatter(
            self.X[labels == -1, 0],
            self.X[labels == -1, 1],
            color="g",
            marker="*",
            label="unlabeled",
        )
        if legend:
            plt.legend(scatterpoints=1, shadow=False, loc="upper right")
        plt.title(title)
        plt.show()

    def fit(self, alpha=0.8, tolerance=0.001, max_iter=200):
        graph_matrix = kneighbors_graph(self.X, 3, mode='connectivity', include_self=True)

        classes = np.array([0, 1])  # class labels for two classes inner and outer

        Y1 = np.zeros((len(self.labels), len(classes)))
        for label in classes:
            Y1[self.labels == label, classes == label] = 1

        Y0 = np.copy(Y1) * (1 - alpha)
        Y_previous = np.zeros((self.X.shape[0], len(classes)))

        for _ in range(max_iter):
            if np.abs(Y1 - Y_previous).sum() < tolerance:
                break
            Y_previous = Y1
            Y1 = graph_matrix @ Y1
            Y1 = alpha * Y1 + Y0

        outLabels = np.zeros(Y1.shape[0])
        for i in range(Y1.shape[0]):
            if Y1[i, 0] == 0 and Y1[i, 1] == 0:
                outLabels[i] = -1
            else:
                outLabels[i] = classes[np.argmax(Y1[i])]

        return outLabels



graphLabelProp = GraphLabelPropagation(X, labels, outer, inner)


graphLabelProp.plotScatter(labels, title="Initial circles", outerTitle="Outer labeled", innerTitle="inner labeled")


for i in range(5, 105, 5):
    learned_labels = graphLabelProp.fit(max_iter=i)
    graphLabelProp.plotScatter(learned_labels, title=f"at iteration = {i}", legend=False)


