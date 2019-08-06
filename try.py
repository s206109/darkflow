# imort YOLOs function
from darkflow.net.build import TFNet
from darkflow.utils import box
from darkflow.utils.pascal_voc_clean_xml_evaluation import pascal_voc_clean_xml
from darkflow.utils import process

# Prepare the environment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import re
import math

from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

################################################################
# parameters
################################################################
visualPath = 'visualization'

labels = ['car','negative']
threshold = 0.7
_, meta = process.parser('cfg/tiny-yolo-kitti-3d-10.cfg')



################################################################
#extract annotations data
################################################################
print('extract annotations data')
gtBoxes = pascal_voc_clean_xml('data/kitti/set1/AnnotationsTrain', labels, exclusive = False)


resultDF = pd.DataFrame(columns = ['gw','gh','gx','gy','gz','ga'])
for dInd in np.arange(0,len(gtBoxes)): #dInd = 何ファイル目なのかの数
        for gInd in np.arange(1,len(gtBoxes[dInd])):
            Width    = (gtBoxes[dInd][gInd][3] - gtBoxes[dInd][gInd][1])/1242
            Height   = (gtBoxes[dInd][gInd][4] - gtBoxes[dInd][gInd][2])/375

            cellX = 1. * 13 / 1242 #画像の横幅を１グリッドあたりのピクセル数
            cellY = 1. * 13 / 375 #画像の縦幅１グリッドあたりのピクセル数

            centerX  =  .5*(gtBoxes[dInd][gInd][3] + gtBoxes[dInd][gInd][1])
            centerY  =  .5*(gtBoxes[dInd][gInd][4] + gtBoxes[dInd][gInd][2])

            centerX  = centerX / cellX
            centerY  = centerY / cellY

            Distance =  gtBoxes[dInd][gInd][5]/100
            Alpha    =  gtBoxes[dInd][gInd][6]

            resultDF = resultDF.append(pd.Series([ Width, Height, centerX, centerY, Distance, Alpha],
                       index=resultDF.columns),ignore_index=True)

################################################################
#reshape annotations data
################################################################
total_features2 = resultDF[["gx","gy","gw","gh","ga"]].as_matrix()
total_dist2   = resultDF[["gz"]].as_matrix()

# Keep  samples for training

train_features2 = total_features2[:20000]
train_dist2 = total_dist2[:20000]


# Keep samples for validation
valid_features2 = total_features2[20000:25000]
valid_dist2 = total_dist2[20000:25000]

# Keep remaining samples as test set
test_features2 = total_features2[25000:]
test_dist2 = total_dist2[25000:]


################################################################
#construct network model
################################################################

nb_obs = total_features2.shape[0]
print("There is {} observations in our dataset ".format(nb_obs))

nb_feature = total_features2.shape[1]
print("There is {} features in our dataset ".format(nb_feature))

import pdb; pdb.set_trace()
nb_hidden = 5

# Set model weights - with random initialization
W1 = tf.Variable(tf.truncated_normal([nb_feature, nb_hidden], mean=0.0, stddev=1.0, dtype=tf.float64), name="weight1")
W2 = tf.Variable(tf.truncated_normal([nb_hidden, 1], mean=0.0, stddev=1.0, dtype=tf.float64), name="weight2")
b1 = tf.Variable(tf.zeros(nb_hidden, dtype = tf.float64), name="bias1")
b2 = tf.Variable(tf.zeros(1, dtype = tf.float64), name="bias2")


# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

import pdb; pdb.set_trace()
def linear_reg(x,y):
    # Define your equation Ypred = X * W + b
    hidden = tf.add(b1,tf.matmul(x,W1))
    hidden = tf.nn.relu(hidden)
    Ypred = tf.add(b2,tf.matmul(hidden,W2))
    Ypred = tf.nn.sigmoid(Ypred)

    # Define your loss function
    error = tf.reduce_mean(tf.square(y - Ypred))

    # Return values
    return([Ypred,error])


y, cost = linear_reg(train_features2, train_dist2)

# Define your parameter :
learning_rate = 0.01
epochs = 50000
cost_history = [[], []]

# Use gradient descent to minimize loss
optim = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

################################################################
#sess run
################################################################
with tf.Session() as sess:
    sess.run(init)
    for i in list(range(epochs)):
        sess.run(optim) # Execute the gradient descent, according our learning_rate and our cost function

        # For each 10 epochs, save costs values - we can plot it later
        if i % 10 == 0.:
            cost_history[0].append(i+1)
            cost_history[1].append(sess.run(cost))
        if i % 100 == 0:
            print("Cost = ", sess.run(cost))


    valid_cost = linear_reg(valid_features2, valid_dist2)[1]
    print('Validation error =', sess.run(valid_cost))
