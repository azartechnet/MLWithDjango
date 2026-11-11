import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Step 1:Load traning dataset
train_data=keras.utils.image_dataset_from_directory("dataset/train",image_size=(128,128),batch_size=32,label_mode="binary")

#step 2:Load validation dataset
val_data=keras.utils.image_dataset_from_directory("dataset/validation",image_size=(128,128),batch_size=32,label_mode="binary")

#step 3:create CNN Model

model=keras.Sequential([
    layers.Input(shape=(128,128,3))
])

#Normalize Images

layers.Rescaling(1./255)

#Convolution layer

layers.Conv2D(32,(3,3),activation='relu')

#Pooling layer

layers.MaxPooling2D((2,2)),

#Second Convolution layer

layers.Conv2D(64,(3,3),activation='relu'),

layers.MaxPooling2D((2,2)),



