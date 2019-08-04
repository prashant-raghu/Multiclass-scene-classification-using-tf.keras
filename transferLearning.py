import tensorflow as tf
print(tf.__version__)
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
SIZE = (150, 150, 3)
# Create the base model from the pre-trained model MobileNet V2 which is trained on imagenet Dataset
# which consists of 1.2M labelled images.
imagenet = tf.keras.applications.MobileNetV2(input_shape=SIZE,
                                               include_top=False,
                                               weights='imagenet')
#To tell Tf not to adjust weigths of imagenet model which are already trained
imagenet.trainable = False
#model Summary
imagenet.summary()
#Defining New Model with imagenet as the Base Model using Sequential Api
model = tf.keras.models.Sequential([
    imagenet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(6, activation='softmax')
])
#Using ImageDataGenerator to ease Data preparation as it lables images based on Folder Name which is ideal for the way Data Set is arranged
TRAINING_DIR = "/tmp/seg_train/"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "/tmp/seg_test/"
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(150, 150))
#Using Adam as it works well with image classification and can adjust Learning rate while training unlike
#GradientDescentOptimizer where manual LR tuning needs to be done.
#Using sparse_categorical_crossentropy as Loss function for similar reasons.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(train_generator,
                              epochs=5,
                              verbose=1,
                              validation_data=validation_generator)
#Saving Model for future Predictions
model.save('TLmodel.h5')
# import matplotlib.image  as mpimg
# import matplotlib.pyplot as plt

# #-----------------------------------------------------------
# # Retrieve a list of list results on training and test data
# # sets for each training epoch
# #-----------------------------------------------------------
# acc=history.history['acc']
# val_acc=history.history['val_acc']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# epochs=range(len(acc)) # Get number of epochs
# #------------------------------------------------
# # Plot training and validation accuracy per epoch
# #------------------------------------------------
# plt.plot(epochs, acc, 'r', "Training Accuracy")
# plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
# plt.title('Training and validation accuracy')
# plt.figure()
# #------------------------------------------------
# # Plot training and validation loss per epoch
# #------------------------------------------------
# plt.plot(epochs, loss, 'r', "Training Loss")
# plt.plot(epochs, val_loss, 'b', "Validation Loss")
# plt.figure()
# # Desired output. Charts with training and validation metrics. No crash :)
