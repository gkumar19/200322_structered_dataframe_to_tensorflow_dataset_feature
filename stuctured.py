# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:36:52 2020

@author: Gaurav
"""


import pandas as pd
import tensorflow as tf

#%% clean the dataframe
df = pd.read_csv('train.csv')
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = df.drop(drop_columns, axis='columns')
df.isna().sum()
fill_age = df['Age'].mean()
df['Age'].fillna(fill_age, inplace=True)
df['Embarked'].fillna('na', inplace=True)


#%% convert into data pipeline
def df_to_ds(df):
    df = df.copy()
    labels = df.pop('Survived')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels)).batch(5)
    return ds

ds = df_to_ds(df)

#%% print some values
for feature, label in ds.take(1):
    print(feature.keys())
    print()
    print(feature['Age'])
    print()
    print(label)

#%% print the layer conversion
def utility_convert(ds, feature_column):
    element = next(iter(ds))[0]
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    return feature_layer(element).numpy()

age = tf.feature_column.numeric_column('Age')
fare = tf.feature_column.numeric_column('Fare')
age_bucket = tf.feature_column.bucketized_column(age, [0, 20, 40, 60, 80])
sex = tf.feature_column.categorical_column_with_vocabulary_list('Sex', ['male', 'female'])
sex_indicate = tf.feature_column.indicator_column(sex)
feature_column = [fare, age_bucket, sex_indicate]

#%%
temp = utility_convert(ds, feature_column)

#%%
ds = tf.data.experimental.make_csv_dataset('train.csv', label_name='Survived', batch_size=1)
print(next(iter(ds.take(1))))

#%%
df['Sex'] = pd.Categorical(df['Sex'])
df['Sex'] = df['Sex'].cat.codes

#%%
df['Embarked'] = pd.Categorical(df['Embarked'])
df['Embarked'] = df['Embarked'].cat.codes

#%%
df_new = df.to_dict('list')