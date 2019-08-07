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

from dynamic_anchor import dynamic_generator



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
total_features2 = resultDF[["gx","gy","gw","gh"]].as_matrix()
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

import pdb; pdb.set_trace()
print(dynamic_generator(test_features2))
