from enum import auto
from aa import VariationalAutoEncoder
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 60

def load_mnist():
    # 60.000 training  -  10.000 testing
    (x_train, y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.astype("float32")/255
    x_train = x_train[...,np.newaxis]
    
    x_test = x_test.astype("float32")/255
    x_test = x_test[...,np.newaxis]

    return x_train,y_train,x_test,y_test

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
# generator function from dataset
# each time this function is called -> return a random sample from data
# def iterate_tracks():
#     for i in range(100):
#         idx =  i # np.random.randint(0,100)
#         yield (x_train[idx],x_train[idx])


# from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import backend as K
if __name__=="__main__":
    x_train,_,_,_ = load_mnist()
    # x_train = tf.data.Dataset.from_generator(generator=iterate_tracks,output_shapes=x_train.shape, output_types=(tf.float64, tf.uint8))
    # x_train = x_train.batch(32)
    
    variational_auto_encoder = train(x_train[:5000],LEARNING_RATE,BATCH_SIZE,EPOCHS)
    variational_auto_encoder.save("model_gen2")
    # autoencoder2 = AutoEncoder.load("model")
    # autoencoder2.summary()
