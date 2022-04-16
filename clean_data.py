#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[14]:


#df = pd.read_csv("dataset.csv")
#print("The shape of df:", df.shape)
#df.head()


# In[11]:


def clean_data(dataframe, 
               solve_missing : str ='drop', 
               solve_missing_cat : str = 'drop',
               drop_list : list = [], 
               category_list : list = []): 
    
    #cleaning the data before analysis
    #drop_list: the columns should be removed
    
    df = dataframe.drop_duplicates() #去除重复项
    df = df.drop(drop_list, axis=1) #drop the id columns which is useless
    if solve_missing == 'drop' or solve_missing_cat == 'drop':
        df = df.dropna() #删除所有包含NaN的行
        object_dtype = list(df.select_dtypes(include='object').columns) #选出类别column
        for col in df.columns:
            if col in category_list or col in object_dtype:
                df[col] = df[col].astype('category')
    else:
        object_dtype = list(df.select_dtypes(include='object').columns)
        for col in df.columns:
            if col in category_list or col in object_dtype:
                df[col] = df[col].astype('category') #把所有category项转变数据类型
                if solve_missing_cat == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0], inplace=False) #用col里的众数对缺失数据进行填充
            else:
                if solve_missing == 'mean':
                    df[col] = df[col].fillna(df[col].mean(), inplace=False) #用col里的mean对缺失数据进行填充
                else:
                    df[col] = df[col].fillna(df[col].median(), inplace=False) #用col里的median对缺失数据进行填充
    return df

def split_xy(dataframe, label : str):
    # split the orginal df into x data and y data
    return dataframe.drop(labels=label, axis=1), dataframe[label]

def encode(df):
    #change the category into dummy variables
    category_dtype = list(df.select_dtypes(include='category').columns)
    cat = pd.get_dummies(df, columns = category_dtype, drop_first = True) #change the category into dummy variables
    for col in cat.columns:
        if cat[col].dtype == np.uint8:
            cat[col] = cat[col].astype('category')
    return cat

def scale(features : tuple):
    trainFeatures, valFeatures, testFeatures = features # type pandas DataFrame
    scaler = StandardScaler()
    category_dtype = list(trainFeatures.select_dtypes(include='category').columns)
    continuous_dtype = list(filter(lambda c: c not in category_dtype, trainFeatures.columns))
    # SCALING
    scaler.fit(trainFeatures[continuous_dtype])
    cont_xtrain = scaler.transform(trainFeatures[continuous_dtype])
    cont_xval = scaler.transform(valFeatures[continuous_dtype])
    cont_xtest = scaler.transform(testFeatures[continuous_dtype])
    # ENCODING
    cat_xtrain = trainFeatures[category_dtype]
    cat_xval = valFeatures[category_dtype]
    cat_xtest = testFeatures[category_dtype]
    #print(cat_xtrain.shape, cat_xval.shape, cat_xtrain.shape)
    xtrain = np.concatenate((cont_xtrain, cat_xtrain), axis=1)
    xval = np.concatenate((cont_xval, cat_xval), axis=1)
    xtest = np.concatenate((cont_xtest, cat_xtest), axis=1)
    return scaler, xtrain, xval, xtest


# In[12]:


def processing_data(df, val_size = 0.2, test_size = 0.3):
    
    #The whole processing of the data
    
    TARGET = 'hospital_death'
    unique_labels = np.unique(df[TARGET])
    category_list = ['elective_surgery', 'ethnicity', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem', 
                 'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis', 'hospital_death']
    drop_list = ['encounter_id', 'patient_id', 'hospital_id', 'Unnamed: 83']
    clean_df = clean_data(df, 'mean', 'mode', drop_list, category_list)
    
    features, labels = split_xy(clean_df, TARGET)
    VAL_SIZE = val_size
    TEST_SIZE = test_size
    encoded_features = encode(features)
    #print('Encoded Features Shape : {}'.format(encoded_features.shape))
    # split all samples to train and test with respect to the ratio of each class
    X_train, X_test, ytrain, ytest = train_test_split(encoded_features, labels, test_size = (VAL_SIZE+TEST_SIZE), stratify = labels)
    # split test sample to test and validation with respect to the ratio of each class
    X_test, X_val, ytest, yval = train_test_split(X_test, ytest, test_size = VAL_SIZE/(VAL_SIZE+TEST_SIZE), stratify = ytest)
    # scale continous variables with standard scaler
    scaler, xtrain, xval, xtest = scale((X_train, X_val, X_test))
    #print(scaler.mean_)
    #print(scaler.var_)
    #print(xtrain.shape, ytrain.shape)
    #print(xval.shape, yval.shape)
    #print(xtest.shape, ytest.shape)
    # convert features to numpy array format
    X_train = np.array(xtrain, dtype='float32')
    X_val = np.array(xval, dtype='float32')
    X_test = np.array(xtest, dtype='float32')
    y_train = np.array(ytrain, dtype='float32')
    y_val = np.array(yval, dtype='float32')
    y_test = np.array(ytest, dtype='float32')
    
    return X_train, X_val, X_test, y_train, y_val, y_test

