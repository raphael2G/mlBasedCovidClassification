from load_data import create_data_split
from train_model import train_model

import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers
from keras.models import Sequential
from vit_tensorflow.vit import ViT

import tensorflowjs as tfjs

import os
from PIL import Image
import numpy as np

from matplotlib import pyplot as plt



#1. get data 

# 0 is not covid, 1 is covid
class_names = ['CT_NonCOVID', 'CT_COVID']

non_covid_training_file_name_path = 'COVID-CT/Data-split/NonCOVID/trainCT_NonCOVID.txt'
covid_training_file_name_path = 'COVID-CT/Data-split/COVID/trainCT_COVID.txt'

non_covid_validation_file_name_path = 'COVID-CT/Data-split/NonCOVID/valCT_NonCOVID.txt'
covid_validation_file_name_path = 'COVID-CT/Data-split/COVID/valCT_COVID.txt'

#2. create model





#3. train model
train_ds = create_data_split(non_covid_training_file_name_path, covid_training_file_name_path)
validation_ds = create_data_split(non_covid_validation_file_name_path, covid_validation_file_name_path)


# #load data -- horses or humans data
# train_ds = tfds.load('horses_or_humans', split='train', as_supervised=True, shuffle_files=True).take(10)
# validation_ds = tfds.load('horses_or_humans', split='test', as_supervised=True, shuffle_files=True)


# #load data -- penguin data
# ds_split, info = tfds.load("penguins/processed", split=['train[:20%]', 'train[20%:]'], as_supervised=True, with_info=True)
# validation_ds = ds_split[0]
# train_ds = ds_split[1]

assert isinstance(train_ds, tf.data.Dataset)

print('Train Datset: %i' %len(train_ds))
try: 
    print('Test Datset: %i' %len(validation_ds))
except: 
    print('No Test Dataset')

model = ViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

train_model(model, n_epochs=201, train_ds=train_ds, validation_ds=validation_ds, batch_size=32, update_increment=1)

