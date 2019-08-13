# imort YOLOs function
#from darkflow.net.build import TFNet
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


# Prepare the environment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import re
import math
import sys

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
cfgPath    = 'cfg/tiny-yolo-kitti-3d-10-dynamic.cfg'
labels = ['car','negative']
threshold = 0.7
_, meta = process.parser(cfgPath)
W, H, anc_num = 13, 13, int(cfgPath[23:25])

################################################################
# prepare the features
################################################################

features_anchors =  np.reshape(meta['anchors'],[2, 10]) / 13
final = np.zeros([ W, H, 3 , anc_num ]) #final dynamic anchor
dynamic_features = np.zeros([anc_num, 4]) #generatorに渡すための特徴量

################################################################
# calculate dist anchor every cells
################################################################


for ind in range(13): #cellx
    for inda in range(13): #celly
        for indb in range(anc_num): #num of anchors
            final[ind][inda][0][indb] = 13 * features_anchors[0][indb] #cellx number
            final[ind][inda][1][indb] = 13 * features_anchors[1][indb] #celly number

            dynamic_features[indb][2] = features_anchors[0][indb] #cellx number
            dynamic_features[indb][3] = features_anchors[1][indb] #celly number

            dynamic_features[indb][0] = ind  / 13 #cellx number
            dynamic_features[indb][1] = inda / 13 #celly number

        dist_anchors = dynamic_generator(dynamic_features)[0]

        for indc in range(anc_num):
            final[ind][inda][2][indc] = 13 * dist_anchors[indc]

#np.set_printoptions(threshold=np.inf)
print(final)
import pdb; pdb.set_trace()
#np.savetxt('dynamic_anchor.txt', final)
np.savetxt(sys.stdout.buffer, final)
