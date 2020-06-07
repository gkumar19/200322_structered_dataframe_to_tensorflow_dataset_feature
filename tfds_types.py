# -*- coding: utf-8 -*-
"""
"""

import tensorflow as tf
import tensorflow_datasets as tfds

#%%
#ds is just like generator, and can take iter or list
ds = tfds.load(name="mnist", download=False, split='train')
len(list(ds))
ds_list = list(ds)
ds_take = ds.take(1)
len(list(ds_take))

ds, info = tfds.load(name="mnist", download=False, split='train', with_info=True)
ds, info = tfds.load(name="mnist", download=False, with_info=True)

#%%

isinstance(ds, tf.data.Dataset)
print(ds)

iterator = iter(ds)
temp = next(iterator)
image = temp['image'].numpy()

#%%
ds_survised = tfds.load(name="mnist", download=False, split='train',as_supervised=True)
ds_unsurvised = tfds.load(name="mnist", download=False, split='train',as_supervised=False)

#%%
ds = tfds.load(name="mnist", download=False, split='train[:1%]')
ds = tf
tfds.list_builders()

#%%
def gen(i):
    for k in range(i):
        yield k

iterate = gen(5) #generator

x = [1,2, 3, 4]


next(iterate)
iter_list = list(iterate)
print(next(iterate))

#%%
ds1 = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
ds2 = tf.data.Dataset.from_tensor_slices([5, 6, 7, 8])
ds = tf.data.Dataset.zip((ds1, ds2))

