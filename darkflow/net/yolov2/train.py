import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf
from ..yolo.misc import show
import numpy as np
import os
import math
import pdb

def expit_tensor(x): #
	return 1. / (1. + tf.exp(-x))

def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta = cfg

    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W # number of grid cells
    anchors = m['anchors']
    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]
    size3 = [None, HW, B, 1]

    # return the below placeholders

    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    _dista = tf.placeholder(tf.float32, size3)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    self.placeholders = {
        'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright, 'dista':_dista
    }

    # Extract the coordinate prediction from net.out
    anchors = np.reshape(anchors, [1, 1, B, 3]) #他に合うようにリシェイプ
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C + 1)])#１３x１３x１０x８ 座標４＋信頼度１＋距離１＋クラス２
    coords = net_out_reshape[:, :, :, :, :4]# 座標の４まで.-1を指定した次元は削除される
    coords = tf.reshape(coords, [-1, H*W, B, 4]) #セルxセルをセル番号
    distance = net_out_reshape[:, :, :, :, 7]# distance
    distance = tf.reshape(distance, [-1, H*W, B, 1])
    adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])#シグモイド関数にかける
    adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * anchors[:,:,:,0:2] / np.reshape([W, H], [1, 1, 1, 2]))
    adjusted_distance_z = tf.sqrt(tf.exp(distance[:,:,:,:1]) * anchors[:,:,:,2:] / np.reshape([W], [1, 1, 1, 1])) #適当にロスっぽくしてみる
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3) #こいつらを繋げる
    import pdb; pdb.set_trace()
    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4]) #
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:7])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])
    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)
    adjusted_net_out = tf.concat([adjusted_net_out, adjusted_distance_z], 3)
    import pdb; pdb.set_trace()
    #↓coordsの要素を二乗xセルの数(13)
    wh = tf.pow(coords[:,:,:,2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    area_pred = wh[:,:,:,0] * wh[:,:,:,1]
    centers = coords[:,:,:,0:2]
    floor = centers - (wh * .5)
    ceil  = centers + (wh * .5)

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil , _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

    # calculate the best IOU, set 0.0 confidence for worse boxes

    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro
    weight_dis = tf.concat(1 * [tf.expand_dims(confs, -1)], 3)
    disid =  0.0 * weight_dis


    self.fetch += [_probs, confs, conid, cooid, proid, disid, _dista]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs, _dista], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid, disid], 3)

    print('Building {} loss'.format(m['model']))
    loss = tf.pow(adjusted_net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)
    import pdb; pdb.set_trace()
