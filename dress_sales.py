# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:38:19 2017

@author: Korah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import os

datapath = "C:\\Users\\Korah\\Documents\\Data Science\\PGDBA\\1st Sem\\Accubits\\Dresses_Attribute_Sales\\"
sales = pd.read_excel(datapath + 'Extra Files\\Dress Sales.xlsx')
attributes = pd.read_excel(datapath + 'Attribute Dataset.xlsx')

sales.head()
sales_copy = sales.copy()
sales.isnull().sum()
to_datetime(sales_copy.iloc[0, 1:])


# EDA
sales_copy.head()
sales_dress_id = sales_copy.iloc[:,0]
sales_num = sales_copy.iloc[:,1:]
sales_num.head()
sales_num.columns = pd.to_datetime(sales_num.columns)
sales_num.columns.astype('str')

sales_dress_id_1 = pd.DataFrame(sales_dress_id)
sales_dress_id_1.head()
sales_copy = pd.merge(sales_dress_id_1, sales_num)
print(sales_copy.columns[3:5].to_str())

# Clustering the attributes dataset
attributes.head()

attrib_features = attributes.drop('Dress_ID', axis = 1)
attrib_features['Season_new'] =  [map(lambda x: x.title(), attrib_features['Season'])]

attrib_features['Season']  = attrib_features['Season'].title()
from sklearn.cluster


# Imputing missing values
def impute_missing(X):
        imputer = Imputer(missing_values= 'NaN', strategy = 'mean', axis = 1)
        imputer = imputer.fit(X.iloc[:,1:])
        X.iloc[:,1:] = imputer.transform(X.iloc[:, 1:])
        
        
k = impute_missing(sales)
