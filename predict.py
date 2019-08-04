import numpy as np
import tensorflow as tf
from tensorflow import keras

image = keras.preprocessing.image
model = keras.models.load_model('modelsaved.h5')
#path to any image to be predicted
path = '/tmp/seg_pred/' + '233' + '.jpg'
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#[x] can be an array of images 
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes)
# Desired output. Charts with training and validation metrics. No crash :)