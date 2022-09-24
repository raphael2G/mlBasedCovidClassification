import tensorflowjs as tfjs
import tensorflow as tf
from PIL import Image

def convert_and_save(model, tfjs_target_dir):
      print(model.summary())
      tfjs.converters.save_keras_model(model, tfjs_target_dir)
      print('-- -- -- -- MODEL CONVERTED -- -- -- --')

