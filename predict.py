import numpy as np
import tensorflow as tf
from tensorflow import keras
#from keras.preprocessing import image
image = keras.preprocessing.image
import random
print(tf.__version__)

# pred = model.predict(test_images)
# print(pred[0])
model = keras.models.load_model('modelsaved.h5')
#for x in range(1,7):
path = '/tmp/seg_pred/' + '233' + '.jpg'
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes)
# Desired output. Charts with training and validation metrics. No crash :)