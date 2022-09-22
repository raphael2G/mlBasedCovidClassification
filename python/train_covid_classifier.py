from load_data import create_data_split
from train_model import train

import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from vit_tensorflow.vit import ViT

import tensorflowjs as tfjs

import os
from PIL import Image
import numpy as np

from matplotlib import pyplot as plt
from datetime import datetime

start_time = datetime.now()
end_time = datetime.now()
elapsed = start_time - end_time
print('minutes: ' , elapsed.total_seconds()/60)



#1. get data 

# 0 is not covid, 1 is covid
class_names = ['CT_NonCOVID', 'CT_COVID']

non_covid_training_file_name_path = 'COVID-CT/Data-split/NonCOVID/trainCT_NonCOVID.txt'
covid_training_file_name_path = 'COVID-CT/Data-split/COVID/trainCT_COVID.txt'

non_covid_validation_file_name_path = 'COVID-CT/Data-split/NonCOVID/valCT_NonCOVID.txt'
covid_validation_file_name_path = 'COVID-CT/Data-split/COVID/valCT_COVID.txt'

#2. create model

model = ViT(
    image_size = 300,
    patch_size = 30,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

#3. train model

#3 train the model

# train_ds = create_data_split(non_covid_training_file_name_path, covid_training_file_name_path)
# validation_ds = create_data_split(non_covid_validation_file_name_path, covid_validation_file_name_path)

train_ds = tfds.load('horses_or_humans', split='train', as_supervised=True, shuffle_files=True)
validation_ds = tfds.load('horses_or_humans', split='test', as_supervised=True, shuffle_files=True)

print('train dataset: %i' %len(train_ds))
print('validation dataset: %i' %len(validation_ds))
#3 train the model

train(model, train_ds, validation_ds, n_epochs=50, batch_size=16)