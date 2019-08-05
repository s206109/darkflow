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




################

# parameters
visualPath = 'visualization'

labels = ['car','negative']
threshold = 0.7
_, meta = process.parser('cfg/tiny-yolo-kitti-3d-10.cfg')

print('extract annotations data')
gtBoxes = pascal_voc_clean_xml('data/kitti/set1/Annotations', labels, exclusive = False)

resultDF = pd.DataFrame(columns = ['gw','gh','gz','ga'])
for dInd in np.arange(0,len(gtBoxes)): #dInd = 何ファイル目なのかの数
        for gInd in np.arange(1,len(gtBoxes[dInd])):
            resultDF = resultDF.append(pd.Series([gtBoxes[dInd][gInd][3]/1242, gtBoxes[dInd][gInd][4]/375, gtBoxes[dInd][gInd][5]/100, gtBoxes[dInd][gInd][6]],
                           index=resultDF.columns),ignore_index=True)

import pdb; pdb.set_trace()

################






import os
print(os.listdir("./data"))

boston = load_boston()
# The True passed to load_boston() lets it know that we want features and prices in separate numpy arrays.
print("Shape of design (feature) matrix : \n ", boston.data.shape)
print("List of features : \n ", boston.feature_names)

bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
print("Simples statistics : \n ", bos.describe())

"""
Correlation

corr= bos.corr(method='pearson')

# 1. HeatMap with Seaborn
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)

# 2. If you want to figures out
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]
corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())


#Pair plot

sns.pairplot(bos)
"""

# Get the data
total_features, total_prices = load_boston(True)
total_features2 = resultDF[["gw","gh"]].as_matrix()
total_prices2   = resultDF[["gz"]].as_matrix()

# Keep 300 samples for training
train_features = scale(total_features[:300])
train_features2 = scale(total_features2[:20000])
train_prices = total_prices[:300]
train_prices2 = total_prices2[:20000]


# Keep 100 samples for validation
valid_features = scale(total_features[300:400])
valid_features2 = scale(total_features2[20000:25000])
valid_prices = total_prices[300:400]
valid_prices2 = total_prices2[20000:25000]

# Keep remaining samples as test set
test_features = scale(total_features[400:])
test_features2 = scale(total_features2[25000:])
test_prices = total_prices[400:]
test_prices2 = total_prices2[25000:]


nb_obs = total_features2.shape[0]
print("There is {} observations in our dataset ".format(nb_obs))

nb_feature = total_features2.shape[1]
print("There is {} features in our dataset ".format(nb_feature))

import pdb; pdb.set_trace()
# Set model weights - with random initialization
W = tf.Variable(tf.truncated_normal([nb_feature, 1],
                                    mean=0.0,
                                    stddev=1.0,
                                    dtype=tf.float64),
                name="weight")
# Set model biais - initialized to 0
b = tf.Variable(tf.zeros(1, dtype = tf.float64), name="bias")


# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")


def linear_reg(x,y):
    # Define your equation Ypred = X * W + b
    Ypred = tf.add(b,tf.matmul(x,W))

    # Define your loss function
    error = tf.reduce_mean(tf.square(y - Ypred))

    # Return values
    return([Ypred,error])
import pdb; pdb.set_trace()
y, cost = linear_reg(train_features2, train_prices2)



# Define your parameter :
learning_rate = 0.01
epochs = 2000
cost_history = [[], []]

# Use gradient descent to minimize loss
optim = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


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

    # Plot costs values
    """
    plt.plot(cost_history[0], cost_history[1], 'r--')
    plt.ylabel('Costs')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(visualPath,'new.png'))
    plt.show()
    """

    train_cost = linear_reg(train_features2, train_prices2)[1]
    print('Train error =', sess.run(train_cost))
    valid_cost = linear_reg(valid_features2, valid_prices2)[1]
    print('Validation error =', sess.run(valid_cost))
