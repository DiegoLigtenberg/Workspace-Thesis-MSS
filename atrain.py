from enum import auto
from aa import VariationalAutoEncoder
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
import os

LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 60

LOAD_SPECTROGRAMS_PATH = "train"

def load_mnist():
  # 60.000 training  -  10.000 testing
  (x_train, y_train),(x_test,y_test) = mnist.load_data()
  x_train = x_train.astype("float32")/255
  x_train = x_train[...,np.newaxis]
  
  x_test = x_test.astype("float32")/255
  x_test = x_test[...,np.newaxis]

  return x_train,y_train,x_test,y_test

def load_fsdd(spectrograms_path):
  x_train = []
  y_train = []
  sub = ["mixture","vocals","bass","drums","other","accompaniment"]
  for root,_,file_names in os.walk(spectrograms_path):
    for file_name in file_names:
      file_path = os.path.join(root,file_name)
      
      #mixture
      if sub[0] in file_path:
        normalized_spectrogram = np.load(file_path)
        x_train.append(normalized_spectrogram)
      #vocals
      if sub[1] in file_path:
        normalized_spectrogram = np.load(file_path)
        y_train.append(normalized_spectrogram)
  x_train = np.array(x_train)
  y_train = np.array(y_train)      
  print(x_train.shape,y_train.shape)
  return x_train,y_train







def train(x_train,learning_rate,batch_size,epochs):
  variatonal_auto_encoder = VariationalAutoEncoder(
    input_shape=(28, 28, 1),
    conv_filters=(32, 64, 64, 64),
    conv_kernels=(3, 3, 3, 3),
    conv_strides=(1, 2, 2, 1),
    latent_space_dim=2)

  variatonal_auto_encoder.summary()
  variatonal_auto_encoder.compile(learning_rate)
  variatonal_auto_encoder.train(x_train,batch_size,epochs)
  return variatonal_auto_encoder




#load dataset
x_train,_,_,_ = load_mnist()

from tensorflow.keras import backend as K
if __name__=="__main__":
    # x_train,_,_,_ = load_mnist()    
    # variational_auto_encoder = train(x_train[:5000],LEARNING_RATE,BATCH_SIZE,EPOCHS)
    # variational_auto_encoder.save("model_gen2")
    load_fsdd(LOAD_SPECTROGRAMS_PATH)
    pass