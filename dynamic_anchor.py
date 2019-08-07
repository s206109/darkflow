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
import os



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

def re_dynamic(x):
    # Define your equation Ypred = X * W + b
    hidden = tf.add(_B1,tf.matmul(x,_W1))
    hidden = tf.nn.relu(hidden)
    Ypred = tf.add(_B2,tf.matmul(hidden,_W2))
    Ypred = tf.nn.sigmoid(Ypred)


    # Return values
    return([Ypred])


def dynamic_generator(features):
    result = sess.run(re_dynamic(features))
    return(result)
