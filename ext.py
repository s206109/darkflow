import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

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

# WとBをプリント
print(sess.run('weight1:0'))
print(sess.run('bias1:0'))
print(sess.run('weight2:0'))
print(sess.run('bias2:0'))
