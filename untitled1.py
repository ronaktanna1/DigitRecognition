# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:01:34 2017

@author: Ronak
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import idx2numpy
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
import json
import random
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from numpy import array
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    #equivalent to y intercept
    #constant value carried over across matrix math
    return tf.Variable(tf.constant(0.05, shape=[length]))
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
#        layer = tf.nn.local_response_normalization(layer)
    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
def LecunLCN(X, image_shape, threshold=1e-4, radius=7, use_divisor=True):
    """Local Contrast Normalization"""
    """[http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf]"""

    # Get Gaussian filter
    filter_shape = (radius, radius, image_shape[3], 1)

    #self.filters = theano.shared(self.gaussian_filter(filter_shape), borrow=True)
    filters = gaussian_filter(filter_shape)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    # Compute the Guassian weighted average by means of convolution
    convout = tf.nn.conv2d(X, filters, [1,1,1,1], 'SAME')

    # Subtractive step
    mid = int(np.floor(filter_shape[1] / 2.))

    # Make filter dimension broadcastable and subtract
    centered_X = tf.sub(X, convout)

    # Boolean marks whether or not to perform divisive step
    if use_divisor:
        # Note that the local variances can be computed by using the centered_X
        # tensor. If we convolve this with the mean filter, that should give us
        # the variance at each point. We simply take the square root to get our
        # denominator

        # Compute variances
        sum_sqr_XX = tf.nn.conv2d(tf.square(centered_X), filters, [1,1,1,1], 'SAME')

        # Take square root to get local standard deviation
        denom = tf.sqrt(sum_sqr_XX)

        per_img_mean = tf.reduce_mean(denom)
        divisor = tf.maximum(per_img_mean, denom)
        # Divisise step
        new_X = tf.truediv(centered_X, tf.maximum(divisor, threshold))
    else:
        new_X = centered_X

    return new_X


def gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype = float)
    mid = np.floor(kernel_shape[0] / 2.)
    
    for kernel_idx in range(0, kernel_shape[2]):
        for i in range(0, kernel_shape[0]):
            for j in range(0, kernel_shape[1]):
                x[i, j, kernel_idx, 0] = gauss(i - mid, j - mid)
    
    return tf.convert_to_tensor(x / np.sum(x), dtype=tf.float32)

def gauss(x, y, sigma=3.0):
    Z = 2 * np.pi * sigma ** 2
    return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])

#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
inputs = idx2numpy.convert_from_file(r'C:\Users\Ronak\Desktop\Important\machine-learning-master\projects\digit_recognition\train-images.idx3-ubyte')
labels = idx2numpy.convert_from_file(r'C:\Users\Ronak\Desktop\Important\machine-learning-master\projects\digit_recognition\train-labels.idx1-ubyte')


dataset_size = 2000# just for initial stages
image_height = 28
image_width = 140
dataset = np.ndarray(shape=(dataset_size, image_height, image_width),
                         dtype=np.float32)
print ("dataset:",dataset.shape)
data_labels = []#one-hot labels
data_labels_2=[]#integer labels for display etc
i = 0
w = 0
print ("labels from idx2numpy:",labels.shape)
print ("sample label from idx2numpy:",labels[0])
# need to convert labels to one-hot

lb = preprocessing.LabelBinarizer()
labels_new = lb.fit_transform(labels)
print ("labels from converting idx2numpy to one-hot:",labels_new.shape)
print ("sample label from idx2numpy after conversion to one-hot:",labels_new[0])
while i < dataset_size:
    temp1 = np.hstack([inputs[w], inputs[w + 1], inputs[w + 2], inputs[w + 3], inputs[w + 4]])
    dataset[i, :, :] = temp1
    data_labels.append((labels_new[w], labels_new[w + 1], labels_new[w + 2], labels_new[w + 3], labels_new[w + 4]))
    data_labels_2.append((labels[w], labels[w + 1], labels[w + 2], labels[w + 3], labels[w + 4]))
    if i==0:
        print("One input's shape after concatenation:",temp1.shape)
        print("Corresponding label after concatenation:",data_labels[0])
    w += 5
    i += 1
print ("datatype of labels:",type(data_labels))
#to convert data_labels from list to array type

data_labels=array(data_labels)
print ("datatype of labels after conversion:",type(data_labels))

def displaySequence(n):
    fig=plt.figure()
    plt.imshow(dataset[n])
    plt.show()
    print ('Label : {}'.format(data_labels_2[n]))
displaySequence(random.randint(0, dataset_size))


y = np.asarray(data_labels_2)
X = dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)
print(data_labels[0,1])
displaySequence(0)


n_classes = 11
batch_size = 64
image_size_flat = image_height * image_width
image_shape = (image_height, image_width)
num_channels=1


tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_height, image_width, num_channels))
tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size, 5))

tf_test_dataset = tf.constant(X_test, shape=(len(X_test),image_height, image_width,num_channels))
shape = [batch_size, image_height, image_width, num_channels]




filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

#more filters, featuer map will b
# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36        # There are 36 of these filters.

def neural_network_model(data,keep_prob,shape):
    LCN = LecunLCN(data, shape)
    num_labels=11
    num_hidden=64
    layer_conv1, weights_conv1 = \
    new_conv_layer(input=LCN,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=False)
    layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
        
    reshape, num_hidden = flatten_layer(layer_conv2)
    layer3= new_fc_layer(reshape,num_hidden,20)
    hidden = tf.nn.dropout(layer3, keep_prob)
    num_hidden=20
    logits1 = new_fc_layer(hidden,num_hidden,num_labels,use_relu=True)
    logits2 = new_fc_layer(hidden,num_hidden,num_labels,use_relu=True)
    logits3 = new_fc_layer(hidden,num_hidden,num_labels,use_relu=True)
    logits4 = new_fc_layer(hidden,num_hidden,num_labels,use_relu=True)
    logits5 = new_fc_layer(hidden,num_hidden,num_labels,use_relu=True)

    
    return [logits1,logits2,logits3,logits4,logits5]

def train_neural_network(x):
    
    [logits1, logits2, logits3, logits4, logits5] = neural_network_model(tf_train_dataset,.75, shape)
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits1, tf_train_labels[:,0])) +\
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits2, tf_train_labels[:,1])) +\
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits3, tf_train_labels[:,2])) +\
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits4, tf_train_labels[:,3])) +\
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits5, tf_train_labels[:,4]))
    global_step = tf.Variable(0)
#    learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
    optimizer = tf.train.AdagradOptimizer(0.001).minimize(loss, global_step=global_step)
    num_steps = 10001
    train_prediction=tf.pack([tf.nn.softmax(neural_network_model(tf_train_dataset,1.0, shape)[0]),\
                            tf.nn.softmax(neural_network_model(tf_train_dataset,1.0, shape)[1]),\
                            tf.nn.softmax(neural_network_model(tf_train_dataset, 1.0,shape)[2]),\
                            tf.nn.softmax(neural_network_model(tf_train_dataset, 1.0,shape)[3]),\
                            tf.nn.softmax(neural_network_model(tf_train_dataset, 1.0,shape)[4])])
    test_prediction=tf.pack([tf.nn.softmax(neural_network_model(tf_test_dataset, 1.0,shape)[0]),\
                            tf.nn.softmax(neural_network_model(tf_test_dataset,  1.0,shape)[1]),\
                            tf.nn.softmax(neural_network_model(tf_test_dataset,  1.0,shape)[2]),\
                            tf.nn.softmax(neural_network_model(tf_test_dataset,  1.0,shape)[3]),\
                            tf.nn.softmax(neural_network_model(tf_test_dataset,  1.0,shape)[4])])
    with tf.Session() as session:
        tf.global_variables_initializer().run()  
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
            batch_data = X_train[offset:(offset + batch_size), :, :]
            batch_data= np.reshape(batch_data, (64,28,140,1))
            batch_labels = y_train[offset:(offset + batch_size),:]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
#            print (predictions[step], batch_labels[step,:5])
            if (step % 5 == 0): 
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels[:,:5]))
                print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), y_test[:,:5]))
    

train_neural_network(X_train)