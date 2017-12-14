# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:15:38 2017

@author: Korah
"""
import pandas as pd
from sklearn.model_selection import train_test_split

datapath = "C:\\Users\\Korah\\Documents\\Data Science\\PGDBA\\1st Sem\\Accubits\\Dresses_Attribute_Sales\\"
df = pd.read_excel(datapath + 'Attribute Dataset.xlsx')
# Preliminary Analysis
df.drop('Dress_ID', axis = 1, inplace = True)
df.head()
df.dtypes
df = df.rename(columns = {'waiseline': 'WaistLine'})

'''
# Converting 1st letters of all datapoints to upper-case
df.Style = list(map(lambda x: x.title(), df.Style))
df.Price = df.Price.astype('str')
df.Price = list(map(lambda x: x.title(), df.Price))
df.Size = list(map(lambda x: x.title(), df.Size))
df.Season = df.Season.astype('str')
df.Season = list(map(lambda x: x.title(), df.Season))
df.NeckLine = df.NeckLine.astype('str')
df.NeckLine = list(map(lambda x: x.title(), df.NeckLine))
df.SleeveLength = df.SleeveLength.astype('str')
df.SleeveLength = list(map(lambda x: x.title(), df.SleeveLength))
df.WaistLine = df.WaistLine.astype('str')
df.WaistLine = list(map(lambda x: x.title(), df.WaistLine))
df.Material = df.Material.astype('str')
df.Material = list(map(lambda x: x.title(), df.Material))
df.FabricType = df.FabricType.astype('str')
df.FabricType = list(map(lambda x: x.title(), df.FabricType))
df.Decoration = df.Decoration.astype('str')
df.Decoration = list(map(lambda x: x.title(), df.Decoration))
df['Pattern Type'] = df['Pattern Type'].astype('str')
df['Pattern Type'] = list(map(lambda x: x.title(), df['Pattern Type']))
'''

# Converting data types
df.Style = df.Style.astype('category')
df.Price = df.Price.astype('category')
df.Size = df.Size.astype('category')
df.Season = df.Season.astype('category')
df.NeckLine = df.NeckLine.astype('category')
df.SleeveLength = df.SleeveLength.astype('category')
df.WaistLine = df.WaistLine.astype('category')
df.Material = df.Material.astype('category')
df.FabricType = df.FabricType.astype('category')
df.Decoration = df.Decoration.astype('category')
df['Pattern Type'] = df['Pattern Type'].astype('category')
# Standardizing 
df.isnull().sum()
# There are 266  and 236 null values in Fabric Type and Decoration
# Hence removing them
df.drop(['FabricType', 'Decoration'], axis = 1, inplace = True)
df.Material[1:10]
#df.to_csv('Clean_dress_attributes.csv')

# Cleaning the data



# Imputing Null values
df.isnull().sum()
df_over_10 = df.drop(['WaistLine','Material','Pattern Type'], axis = 1 )
df_over_10 = df_over_10.dropna()
df_over_10.info()

# Removing the nulls
df_nona = df.dropna()
df_nona.head()
df_nona.isnull().sum()
df.info()
df.Recommendation.value_counts()
# 0 - 290
# 1 - 210

'''
# Creating dummy values
df_nona_with_dummy = pd.get_dummies(df_over_10, columns = ['Style','Price','Size','Season','NeckLine','SleeveLength'], 
                                    drop_first = True)
df_nona_with_dummy.columns
df_nona_with_dummy['Rating_1'] = df_nona_with_dummy.Rating
df_nona_with_dummy.drop('Rating', inplace = True, axis = 1)
df_nona_with_dummy.info()
'''
df_over_10 = df.copy()
df_over_10.dtypes
df_over_10.head(20)

# LabelEncoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_label_encoded = df_over_10.apply(le.fit_transform)
df.Style = le.fit_transform(df.Style)
df.Style = df.Style.astype('category')
df.Price = le.fit_transform(df.Price.astype(str))
df.Price = df.Price.astype('category')
df.Size= le.fit_transform(df.Size)
df.Size = df.Size.astype('category')
df.Season= le.fit_transform(df.Season.astype(str))
df.Season = df.Season.astype('category')
df.NeckLine= le.fit_transform(df.NeckLine.astype(str))
df.NeckLine = df.NeckLine.astype('category')
df.SleeveLength= le.fit_transform(df.SleeveLength.astype(str))
df.SleeveLength = df.SleeveLength.astype('category')
df.WaistLine = le.fit_transform(df.WaistLine.astype(str))
df.WaistLine= df.WaistLine.astype('category')
df.Material= le.fit_transform(df.Material.astype(str))
df.Material= df.Material.astype('category')
df['Pattern Type']= le.fit_transform(df['Pattern Type'].astype(str))
df['Pattern Type']= df['Pattern Type'].astype('category')
df.head()
#df.to_csv('df_label_encoded.csv')
df_nona.info()
df_nona = df_over_10.dropna()
df_with_dummy.head()
df_with_dummy = pd.get_dummies(df, columns = ['Style','Price','Size','Season','NeckLine','SleeveLength',
                                                      'WaistLine', 'Material', 'Pattern Type'], 
                                    drop_first = True)
df_with_dummy['Rating_1'] = df_with_dummy.Rating
df_with_dummy.drop('Rating',inplace = True, axis =1)
df_with_dummy.dtypes
df_with_dummy.to_csv('df_500_imputed_with_dummy.csv')
df.dtypes
'''
#Cleaning Each attribute
df.Style.value_counts()
df.Style = df.Style.replace("sexy","Sexy")
df.Price.value_counts()
df.Price = df.Price.replace("low","Low")
df.Price = df.Price.replace("high","High")
df.Size.value_counts()
df.Size = df.Size.replace("small","S")
df.Size = df.Size.replace("s","S")
df.Season.value_counts()
df.Season = df.Season.replace("spring","Spring")
df.Season = df.Season.replace("summer","Summer")
df.Season.value_counts()
df.Season = df.Season.replace("winter","Winter")
df.Season = df.Season.replace("Automn","Autumn")
df.NeckLine.value_counts()
df.NeckLine = df.NeckLine.replace("sweetheart","Sweetheart")
df.SleeveLength.value_counts()
df.SleeveLength = df.SleeveLength.replace("threequater","threequarter")
df['Pattern Type'].value_counts()
df['Pattern Type'] = df['Pattern Type'].replace("leapord","leopard")

df.to_csv('df500_after_rectifying_mistakes.csv')
'''
# Imputing missing values
df_imputed = pd.read_csv('df500_after_imputation.csv')
df_imputed.Style.isnull()
df_with_dummy.dtypes
df = df_imputed.iloc[:,2:13]
y = df_imputed.iloc[:,12]
df.to_csv('df500_after_imputation_10_features.csv')
df.head()
# Creating Train and Test set
X = df_with_dummy.iloc[:, 1:].values
y = df_with_dummy.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf_classifier = RandomForestClassifier(n_estimators= 200, criterion='entropy', 
                                       max_depth =10, oob_score = True, random_state=1,
                                       max_features= 0.33)
# Fitting the model and finding the accuracy

rf_classifier.fit(X,y)
rf_classifier.oob_score_
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print("Test accuracy = ",accuracy_score(y_pred, y_test))
y_pred_train = rf_classifier.predict(X_train)
print("Train accuracy = ",accuracy_score(y_pred_train, y_train))
# Test accuracy - 0.93
# Train accuracy - 0.94
# Data used : df500_imputed_with_dummy

important_features = []
for x,i in enumerate(rf_classifier.feature_importances_):
    if i>np.average(rf_classifier.feature_importances_):
        important_features.append((x))


feature_names = df_nona_with_dummy.iloc[:,1:].columns
important_names = feature_names[important_features]
print(important_names)


#============
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


importances[indices]
#=============================================================================
from sklearn.externals import joblib
joblib.dump(rf_classifier,'rf_classifier.pkl') 

model = joblib.load('rf_model.pkl')
print(model)
data = 'Brief,Average,4.6L,Spring,	o-neck	full,natural,silk	,chiffon,embroidary,	print'
model.predict([data])
