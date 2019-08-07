import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

sess = tf.Session()

# 訓練済みモデルのmetaファイルを読み込み
saver = tf.train.import_meta_graph('./model_ckpt/linear.meta')


# モデルの復元
saver.restore(sess,tf.train.latest_checkpoint('./'))
