# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:28:13 2020

Data: https://www.kaggle.com/mlg-ulb/creditcardfraud

@author: KGU2BAN
"""

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

#%% using tensorflow dataset manager

def convert_to_numpy(element):
    '''convert element into numpy
    '''
    x = element[0]
    y = element[1]
    keys = x.keys()
    values = []
    for value in x.values():
        values.append(value.numpy())
    x = zip(keys, values)
    x=OrderedDict(x)
    y = y.numpy()
    return x, y

def manipulate_element(x, y):
    '''manipulate tensor element
    '''
    x['Time'] = tf.cast(x['Time'], 'float32') * 0.01
    #x = convert_to_numpy(x)
    return (x, y)

ds = tf.data.experimental.make_csv_dataset('creditcard.csv', 10, shuffle=False, label_name='Class')
ds = ds.map(manipulate_element)

csv_iter = iter(ds)
csv_next = next(csv_iter)

manipulated_element = convert_to_numpy(csv_next)

#%% using python generator
def gen():
    for chunk in pd.read_csv('creditcard.csv', chunksize=20):
        yield OrderedDict(dict(chunk))
    
ds = tf.data.Dataset.from_generator(gen, tuple(['float']*31))

csv_iter = iter(ds)
csv_next = next(csv_iter).numpy()

ite = gen()
temp = next(ite)

#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('creditcard.csv')
df.pop('Time')
labels = df.pop('Class')
df['Amount'] = np.log(df['Amount']+ 0.0001)
x_train, x_test, y_train, y_test = train_test_split(df, labels)

scale = StandardScaler()
scale.fit(x_train)

#%%
import time
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Recall, Precision
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives
metrics = [AUC(name='auc'), BinaryAccuracy(name='acc'), Recall(name='recall'), Precision(name='precision'), FalseNegatives(name='fn'), FalsePositives(name='fp'), TrueNegatives(name='tn'), TruePositives(name='tp')]

#%%
from tensorflow.keras.layers import Dense, Lambda

tf.keras.backend.clear_session()
class Model(tf.keras.models.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.mean = [ 4.86891206e-05,  2.41795624e-03, -7.77870260e-04,  1.24860131e-03,
           -1.11488661e-03, -2.68453403e-03,  2.76898750e-03,  2.30381200e-03,
            8.38725114e-04, -6.18474035e-04, -1.17329905e-03, -5.07198986e-04,
            1.80924019e-03, -1.87002264e-03, -2.47406036e-04,  9.81059979e-04,
            2.99254783e-06,  5.63149682e-04,  8.31414594e-04,  3.69230127e-05,
           -5.36862989e-04, -3.94106533e-04, -2.14451825e-04,  3.98136081e-04,
            1.15514675e-04, -3.62776063e-04,  4.67490526e-04,  2.00237266e-04,
            2.92828924e+00]
        
        self.std = [1.95563435, 1.65069673, 1.51343028, 1.41812564, 1.38516667,
           1.3325455 , 1.24184029, 1.18314112, 1.09817896, 1.08704184,
           1.02107617, 1.00199579, 0.99561848, 0.96509921, 0.91595761,
           0.87839105, 0.8524445 , 0.83814739, 0.81443923, 0.77350581,
           0.72874478, 0.72558829, 0.62724647, 0.60446678, 0.52172715,
           0.48152371, 0.40152981, 0.33307414, 2.10178499]
        
        output_init = tf.keras.initializers.Constant(-0.3)
        
        self.lambda_l = Lambda(lambda x: x*self.std + self.mean, name='lambda', dtype='float')
        self.dense1 = Dense(29, name='dense1', input_shape=(29,), activation='relu')
        self.dense2 = Dense(12, name='dense2', activation='relu')
        self.dense3 = Dense(1, name='dense3', activation='sigmoid', bias_initializer=output_init)
    def __call__(self, x, training=True):
        x = self.lambda_l(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

ds_train = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values)).batch(5000)
ds_test = tf.data.Dataset.from_tensor_slices((x_test.values, y_test.values)).batch(5000)
tensorboard = tf.keras.callbacks.TensorBoard(f'logs\{time.strftime("%Y%m%d-%H%M%S")}')
es = tf.keras.callbacks.EarlyStopping(monitor='val_auc',patience=25, min_delta=0.001, restore_best_weights=True)
model = Model()
model.compile('adam', 'binary_crossentropy', metrics=metrics)
class_weigts = {0: (1*213605/213218), 1: (1*213605/387)}
model.fit(ds_train, epochs=200, validation_data=ds_test, callbacks=[tensorboard], class_weight=None)
#%%
from sklearn.metrics import confusion_matrix, roc_curve

roc = roc_curve(y_test, model(x_test.values))
sns.lineplot(roc[0], roc[1])
sns.lineplot(roc[0], roc[2])

plt.figure()
sns.heatmap(confusion_matrix(y_test, model(x_test.values) > 0.5), annot=True , fmt='d')


#%%
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Recall, Precision
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives
metrics = [AUC(name='auc'), BinaryAccuracy(name='acc'), Recall(name='recall'), Precision(name='precision'), FalseNegatives(name='fn'), FalsePositives(name='fp'), TrueNegatives(name='tn'), TruePositives(name='tp')]

ds_train = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values)).batch(5000)
ds_test = tf.data.Dataset.from_tensor_slices((x_test.values, y_test.values)).batch(5000)
tensorboard = tf.keras.callbacks.TensorBoard(f'logs\{time.strftime("%Y%m%d-%H%M%S")}')
es = tf.keras.callbacks.EarlyStopping(monitor='val_auc',patience=25, min_delta=0.001, restore_best_weights=True)
model = Model()
model.compile('adam', 'binary_crossentropy', metrics=metrics)
class_weigts = {0: (1*213605/213218), 1: (1*213605/387)}
model.fit(ds_train, epochs=200, validation_data=ds_test, callbacks=[tensorboard], class_weight=None)
