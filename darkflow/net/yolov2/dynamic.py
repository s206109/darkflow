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

resultDF = pd.DataFrame(columns = ['gx','gy','gw','gh','gz','ga'])
for dInd in np.arange(0,len(gtBoxes)): #dInd = 何ファイル目なのかの数
        for gInd in np.arange(1,len(gtBoxes[dInd])):
            resultDF = resultDF.append(pd.Series([gtBoxes[dInd][gInd][1]/1242,gtBoxes[dInd][gInd][2]/375,gtBoxes[dInd][gInd][3]/1242, gtBoxes[dInd][gInd][4]/375, gtBoxes[dInd][gInd][5]/100, gtBoxes[dInd][gInd][6]],
                           index=resultDF.columns),ignore_index=True)

import pdb; pdb.set_trace()

################






import os
print(os.listdir("./data"))


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
#total_features, total_dists = load_boston(True)
total_features = resultDF[["gx","gy","gw","gh","ga"]].as_matrix()
total_dists   = resultDF[["gz"]].as_matrix()

# Keep 300 samples for training
#train_features = scale(total_features[:300])
train_features = total_features[:20000]
#train_dists = total_dists[:300]
train_dists = total_dists[:20000]


# Keep 100 samples for validation
#valid_features = scale(total_features[300:400])
valid_features = total_features[20000:25000]
#valid_dists = total_dists[300:400]
valid_dists = total_dists[20000:25000]

# Keep remaining samples as test set
#test_features = scale(total_features[400:])
test_features = total_features[25000:]
#test_dists = total_dists[400:]
test_dists = total_dists[25000:]


nb_obs = total_features.shape[0]
print("There is {} observations in our dataset ".format(nb_obs))

nb_feature = total_features.shape[1]
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

src_vec_length = 5
hidden_unit_num = 10
tar_vec_length = 1

#===========================
# レイヤーの関数
# fc layer
def fc_relu(inputs, w, b):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.relu(fc)
	return fc

# fc layer
def fc(inputs, w, b):
	fc = tf.matmul(inputs, w) + b
	return fc

def weight_variable(name,shape):
    return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name,shape):
    return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
#===========================


#===========================
# tensorflowで用いるデータ群作成
input_data = tf.placeholder(shape=(None,None),dtype = tf.float32,name='input_data')
target_data = tf.placeholder(shape=(None,None),dtype = tf.float32,name='target_data')

# 線形回帰で必要なW,Bを作成
W1 = weight_variable('weight1',[src_vec_length,hidden_unit_num])
B1 = weight_variable('bias1',[hidden_unit_num])
W2 = weight_variable('weight2',[hidden_unit_num,hidden_unit_num])
B2 = weight_variable('bias2',[hidden_unit_num])
W3 = weight_variable('weight3',[hidden_unit_num,tar_vec_length])
B3 = weight_variable('bias3',[tar_vec_length])

# 活性化関数を指定してfc層を生成。
fc1 = fc_relu(X, W1, B1)
fc2 = fc_relu(fc1, W2, B2)
fc_out = fc(fc2,W3,B3)
#===========================


# Define your parameter :
learning_rate = 0.01
epochs = 2000
cost_history = [[], []]

#===========================
# loss関数(平均二乗誤差)
loss = tf.reduce_mean(tf.square(Y - fc_out))

# optimizerの設定
train_optimaizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# 初期化
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#===========================




with tf.Session() as sess:
    for i in list(range(epochs)):
        _, loss_train = sess.run([train_optimaizer,loss],feed_dict={X:train_features, Y:train_dists}) # Execute the gradient descent, according our learning_rate and our cost function

        # For each 10 epochs, save costs values - we can plot it later
        if i % 10 == 0.:
            cost_history[0].append(i+1)
            cost_history[1].append(loss_train)
        if i % 100 == 0:
            print("Cost = ",loss_train)

    # Plot costs values
    """
    plt.plot(cost_history[0], cost_history[1], 'r--')
    plt.ylabel('Costs')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(visualPath,'new.png'))
    plt.show()
    """

    loss_valid = sess.run([loss],feed_dict={X:valid_features, Y:valid_dists}) # Execute the gradient descent, according our learning_rate and our cost function
    print('Validation error =', loss_valid)
