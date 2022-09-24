import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


def process_file(file_name, img_size=224):
    img = Image.open(file_name)
    if np.shape(img)[-1] != 3:
        try: 
            shape = np.shape(img)
            img = img.convert('RGB')
            print('converted: ', shape, 'to', np.shape(img))
        except:
            print('Invalid Data Removed: ')
            print(np.shape(img))
            return '', False
    img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    img = np.asarray(img)
    img = img/255.0
    return img, True

def create_data_split(non_covid_file_name_path, covid_file_name_path, items=None, shuffle=False, img_size=224, normalize=False):
    non_covid_file_name_list = open(non_covid_file_name_path).readlines()
    covid_file_name_list = open(covid_file_name_path).readlines()

    numpy_data = []
    numpy_labels = []


    count = 0
    for non_covid_file_name, covid_file_name in zip(non_covid_file_name_list, covid_file_name_list):
        
        non_covid_file_name = non_covid_file_name.strip()
        covid_file_name = covid_file_name.strip()

        non_covid_path = os.path.join('COVID-CT/Images-processed/CT_NonCOVID', non_covid_file_name)
        covid_path = os.path.join('COVID-CT/Images-processed/CT_COVID', covid_file_name)

        non_covid_img, valid = process_file(non_covid_path, img_size)
        if valid: 
            numpy_data.append(non_covid_img)
            numpy_labels.append(0)

        covid_img, valid = process_file(covid_path, img_size)
        if valid: 
            numpy_data.append(covid_img)
            numpy_labels.append(1)

        if items!=None:
            count += 1
            if count == items: 
                break


    tensor_data = tf.convert_to_tensor(numpy_data)
    tensor_labels = tf.convert_to_tensor(numpy_labels)

    dataset = tf.data.Dataset.from_tensor_slices((tensor_data, tensor_labels))

    if shuffle: dataset = dataset.shuffle(len(list(dataset))-1)

    return dataset

