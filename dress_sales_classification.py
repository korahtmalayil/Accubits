# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:35:57 2017

@author: Korah
"""

# Dress Sales Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the dataset
datapath = "C:\\Users\\Korah\\Documents\\Data Science\\PGDBA\\1st Sem\\Accubits\\Dresses_Attribute_Sales\\"
sales = pd.read_excel(datapath + 'Extra Files\\Dress Sales.xlsx')
dataset = pd.read_excel(datapath + 'Attribute Dataset.xlsx')

# Preliminary analysis
attrib_features = dataset.copy
attrib_features.isnull().sum()
attrib_features.head()

(attrib_features)
i =0    
'''for i in [0,1,3,4,5,6,7,8,9,10,12]:
    attrib_features.iloc[:i] = pd.Series(list(map(lambda x: x.title(), attrib_features.iloc[:,i])))
'''
# Coverting all required columns to str    





attrib_features.iloc[:,11] = pd.Series(list(map(lambda x: x.title(), attrib_features.iloc[:,11])))
attrib_features['Pattern Type'] = attrib_features['Pattern Type'].astype('str')

attrib_features['Pattern Type'][attrib_features['Pattern Type'].str.contains('Nan')] = float('nan')
attrib_features = attrib_features.iloc[:,0:13]

from, sklearn.model_selection import train_test_split
X = attrib_features.iloc[:, 0:12].values
Y = attrib_features.iloc[:, 12].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
X[1:5]
Y[1:3]

#==============================================================================
# Dataset cleaning
dataset.drop('Dress_ID', axis = 1, inplace = True)
dataset.head()
dataset_dummy.dtypes
dataset.isnull().sum()
dataset.drop(['Material','FabricType','Decoration','Pattern Type'], axis = 1, inplace = True)

dataset = dataset.rename(columns = {'WasitLine':'WaistLine'})
dataset_dummy = pd.get_dummies(dataset, columns = ['Style','Price','Size','Season','NeckLine',
                                                   'SleeveLength', 'WaistLine','Material',
                                                   'FabricType', 'Decoration', 'Pattern Type'], drop_first= True)
dataset_dummy.columns

X_1 = dataset_dummy.iloc[:,2:].values
#dataset_dummy['Rating_1'] = dataset_dummy.iloc[:,0]
Y_1 = dataset_dummy.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1, random_state = 1)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.apply(le.fit_transform)

# Classification - Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
classifier = RandomForestClassifier(n_estimators = 1000, max_features=4, criterion = 'entropy', random_state= 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
accuracy_score(y_pred, y_test)
from sklearn.feature_selection import RFE
selector = RFE()

# SVM
from sklearn.svm import SVC
sv_classifier = SVC(kernel='rbf', gamma=4)
sv_classifier.fit(X_train, y_train)
y_pred = sv_classifier.predict(X_test)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors= 8)`
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

# Logistic Regression
