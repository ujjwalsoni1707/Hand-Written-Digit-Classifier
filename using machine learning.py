# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:01:00 2020

@author: Ujjwal Soni
"""


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
print("Desciption of Data")
print(dir(digits))
print("")
plt.gray()
plt.matshow(digits.images[35])
#print(digits.target[35])
print("Representation of Data item")
print(digits.data[35])
print("")

#Splitting the data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
model.fit(X_train, y_train)
print("Score of Logistic Regression",model.score(X_test, y_test))
print("Predicted value by Logistice Regression",model.predict(digits.data[[35]]))
print()

#COnfusion Matrix
#y_predicted = model.predict(X_test)
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_predicted)
#import seaborn as sn
#plt.figure(figsize = (10,7))
#sn.heatmap(cm, annot=True)
#plt.xlabel('Predicted')
#plt.ylabel('Truth')

#Decision Tree
from sklearn import tree
model = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, random_state=None,
            splitter='best')
model.fit(X_train, y_train)
print("Score of Decison Tree",model.score(X_test, y_test))
print("Predicted value by Decision Tree",model.predict(digits.data[[35]]))
print()

#Support Vector Machines
from sklearn.svm import SVC
model = SVC(C=2.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape="ovo", degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
model.fit(X_train, y_train)
print("Score of Support Vector Machine",model.score(X_test, y_test))
print("Predicted value by Support Vector Machine",model.predict(digits.data[[35]]))
print()

#Random Forest 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)
print("Score of Random Forest",model.score(X_test, y_test))
print("Predicted value by Random Forest",model.predict(digits.data[[35]]))
print()

#K-Nearest Neigbhours
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("Score of K-Nearest Neighbors",model.score(X_test, y_test))
print("Predicted value by K-Nearest Neighbors",model.predict(digits.data[[35]]))
print()

