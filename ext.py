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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
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

            cellX = 1. * 1242 / 13 #画像の横幅を１グリッドあたりのピクセル数
            cellY = 1. * 375  / 13 #画像の縦幅１グリッドあたりのピクセル数

            centerX  =  .5*(gtBoxes[dInd][gInd][3] + gtBoxes[dInd][gInd][1])
            centerY  =  .5*(gtBoxes[dInd][gInd][4] + gtBoxes[dInd][gInd][2])

            centerX  = centerX / cellX
            centerY  = centerY / cellY

            centerX  = np.floor(centerX) /13
            centerY  = np.floor(centerY) /13

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
Cpoint1 = 10000
Cpoint2 = 13500
#Cpoint1 = 3
#Cpoint2 = 5

train_features2 = total_features2[:Cpoint1]
train_dist2 = total_dist2[:Cpoint1]


# Keep samples for validation
valid_features2 = total_features2[Cpoint1:Cpoint2]
valid_dist2 = total_dist2[Cpoint1:Cpoint2]

# Keep remaining samples as test set
test_features2 = total_features2[Cpoint2:]
test_dist2 = total_dist2[Cpoint2:]


sess = tf.Session()

# 訓練済みモデルのmetaファイルを読み込み
saver = tf.train.import_meta_graph('./model_ckpt/linear.meta')


# モデルの復元
saver.restore(sess,tf.train.latest_checkpoint('./model_ckpt'))

graph = tf.get_default_graph()
_W1 = graph.get_tensor_by_name("weight1:0")
_B1 = graph.get_tensor_by_name("bias1:0")
_W2 = graph.get_tensor_by_name("weight2:0")
_B2 = graph.get_tensor_by_name("bias2:0")

def linear_reg(x,y):
    # Define your equation Ypred = X * W + b
    hidden = tf.add(_B1,tf.matmul(x,_W1))
    hidden = tf.nn.relu(hidden)
    Ypred = tf.add(_B2,tf.matmul(hidden,_W2))
    Ypred = tf.nn.sigmoid(Ypred)

    # Define your loss function
    error = abs(100* (y - Ypred))

    # Return values
    return([Ypred,error])

import pdb; pdb.set_trace()
result = sess.run(linear_reg(test_features2,test_dist2))
print(result[1]) #誤差の平均値

for ind in range(len(result[1])):
    surveyx = ind
    surveyy = result[1][ind]

plt.scatter(surveyx, surveyy,   c='b', s = 5,label = None)

# 凡例を表示する
plt.legend()
plt.xlabel('object index ',fontsize = 18)
plt.ylabel('distance [m]',fontsize = 18)

# グラフのタイトルを設定する
plt.title("Distribution of distance error",fontsize = 18)
plt.savefig(os.path.join(visualPath,'cont_dynamic.png'))
# 表示する
