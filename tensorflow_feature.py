import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import DenseFeatures, Dense
from tensorflow import feature_column

tf.keras.backend.set_floatx('float64')
#%%
df = pd.read_csv('heart.csv')
df['thal'] = df['thal'].str.split('/n').apply(lambda x: x[0])

label_df = df.pop('target')
ds = tf.data.Dataset.from_tensor_slices((dict(df), label_df))
ds = ds.batch(32)

ds = ds.take(3)
for i, j in ds:
    print(j.numpy())
    for key, value in i.items():
        print(key, end='   ')
        print(value.numpy(), end=' ')
        print()


#%%
ds_temp = next(iter(ds))[0]
age = feature_column.numeric_column('age')
feature_layer = DenseFeatures(age)
print(feature_layer(ds_temp))

age_buckets = feature_column.bucketized_column(age,boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_layer = DenseFeatures(age_buckets)
print(feature_layer(ds_temp))

thal = feature_column.categorical_column_with_vocabulary_list('thal', list(df.thal.unique())[:-2])
thal_one_hot = feature_column.indicator_column(thal)
feature_layer = DenseFeatures(thal_one_hot)
print(feature_layer(ds_temp))

thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_layer = DenseFeatures(thal_embedding)
print(feature_layer(ds_temp))

thal_hashed = feature_column.categorical_column_with_hash_bucket(
      'thal', hash_bucket_size=4)
thal_hashed = feature_column.indicator_column(thal_hashed)
feature_layer = DenseFeatures(thal_hashed)
print(feature_layer(ds_temp))


crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=10)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_layer = DenseFeatures(crossed_feature)
print(feature_layer(ds_temp))


#%%
feature_columns = []

for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))

age = feature_column.numeric_column('age')
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

thal = feature_column.categorical_column_with_vocabulary_list('thal',
                                                              ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

thal_embedding = feature_column.embedding_column(thal, dimension=9)
feature_columns.append(thal_embedding)

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.models.Sequential([feature_layer,
                                    Dense(128, 'relu'),
                                    Dense(128, 'relu'),
                                    Dense(1, 'sigmoid')])

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.fit(ds, epochs=2)
model.summary()
