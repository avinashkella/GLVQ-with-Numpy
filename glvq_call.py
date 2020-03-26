# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:56:01 2019

@author: Avinash
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from glvq_numpy import Glvq

prototype_per_class = 3
input_data = load_iris().data
data_label = load_iris().target
epochs = 2
learning_rate = 0.1

clf = Glvq(prototype_per_class)

X_train, X_test, y_train, y_test = train_test_split(input_data,
                                                    data_label,
                                                    test_size=0.3,
                                                    random_state=42)

clf.fit(X_train, y_train, learning_rate, epochs)

y_predict = clf.predict(X_test)