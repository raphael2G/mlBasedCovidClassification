import tensorflowjs as tfjs
import tensorflow as tf
from PIL import Image

def convert_and_save(model, tfjs_target_dir):
      # saves models as model.json file. os.path.join with specific 
      tfjs.converters.save_keras_model(model, tfjs_target_dir)
      
      print('-- -- -- -- MODELS CONVERTED -- -- -- --')

model = tf.keras.models.load_model('savedModels/tf_ViT_covid_classifier/')
print(model.summary())
img = Image.open('COVID-CT/example_data/covid_ct_scan_0-14.jpg')
resized = img.resize((224, 224))
resized.show()
# image_array  = tf.keras.preprocessing.image.img_to_array(resized)


