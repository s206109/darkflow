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
    sdist = float(m['dist_scale'])
    salph = float(m['alpha_scale'])

    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W # number of grid cells
    anchors = m['anchors']

    #import pdb; pdb.set_trace()
    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor, sdist, salph]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]
    size3 = [None, HW, B, 1]

    # return the below placeholders
    conspo = tf.constant(0.70000, shape = [169, 10])
    consne = tf.constant(0.30000, shape = [169, 10])
    constm = tf.constant(-1)
    constp = tf.constant(1)
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    _dista = tf.placeholder(tf.float32, size3)
    #_alpha = tf.placeholder(tf.float32, size3)
    _vecX = tf.placeholder(tf.float32, size3)
    _vecY = tf.placeholder(tf.float32, size3)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])
    #_kayu = tf.placeholder(tf.float32, size3)


    self.placeholders = {
        'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright, 'dista':_dista
    }


    #self.placeholders = {
    #    'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
    #    'areas':_areas, 'upleft':_upleft, 'botright':_botright, 'dista':_dista, 'kayu':_kayu
    #}
    #sasaki  = np.reshape(np.eye(B,B), [1, 1, B, B])
    #sasaki = np.reshape(np.array([np.eye(10,10) for i in range(169)]),[13,13,10,10])
    # Extract the coordinate prediction from net.out
    if m['name'].find('3d')>-1 : print('++++++++++++++++3次元で学習します+++++++++++++++')

    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C + 1 )])#１３x１３x１０x８ 座標４＋信頼度１＋距離１＋角度１＋クラス２

    coords = net_out_reshape[:, :, :, :, :4]# 座標の４まで.-1を指定した次元は削除される
    coords = tf.reshape(coords, [-1, H*W, B, 4]) #セルxセルをセル番号
    distance = net_out_reshape[:, :, :, :, 7]# distance
    distance = tf.reshape(distance, [-1, H*W, B, 1])

    """
    if self.FLAGS.alpha:
         #alpha = net_out_reshape[:, :, :, :, 8]# alpha
         vecX = net_out_reshape[:, :, :, :, 8]
         #vecY = net_out_reshape[:, :, :, :, 9]
         #import pdb; pdb.set_trace()
         vecY = tf.sin(vecX)
         vecX = tf.cos(vecX)
         #alpha = tf.reshape(alpha, [-1, H*W, B, 1])
         vecX = tf.reshape(vecX, [-1, H*W, B, 1])
         vecY = tf.reshape(vecY, [-1, H*W, B, 1])
    """

    import pdb; pdb.set_trace()
    if self.FLAGS.dynamic:
        anchors =  np.reshape(anchors, [1,HW,B,3])
        adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])#シグモイド関数にかける
        adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * anchors[:,:,:,0:2] / np.reshape([W, H], [1, 1, 1, 2]))
        adjusted_distance_z = tf.sqrt(tf.exp(distance[:,:,:,:1]) * anchors[:,:,:,2:3] / np.reshape([W], [1, 1, 1, 1]))
    else:
        adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])#シグモイド関数にかける
        adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * anchors[:,:,:,0:2] / np.reshape([W, H], [1, 1, 1, 2]))
        adjusted_distance_z = tf.sqrt(tf.exp(distance[:,:,:,:1]) * anchors[:,:,:,2:3] / np.reshape([W], [1, 1, 1, 1]))

    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3) #こいつらを繋げる
    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4]) #
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:7])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])

    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)
    adjusted_net_out = tf.concat([adjusted_net_out, adjusted_distance_z], 3)

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
    iou = tf.truediv(intersect, _areas + area_pred - intersect) #要素ごとの商
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True)) #アンカーの中でベストをTrue
    best_box = tf.to_float(best_box) #float型に
    confs = tf.multiply(best_box, _confs) #それぞれかけてベストなボックス以外０に（アンカーの中の１つだけ使う）


    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs #(物体が存在しない確率)＋ 5*(存在する確率)
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro
    weight_dis = tf.concat(1 * [tf.expand_dims(confs, -1)], 3)
    disid =  sdist * weight_dis


    weight_alp = tf.concat(1 * [tf.expand_dims(confs, -1)], 3)
    alpid =  salph * weight_alp
    weight_veX = tf.concat(1 * [tf.expand_dims(confs, -1)], 3)
    veXid =  salph * weight_veX
    weight_veY = tf.concat(1 * [tf.expand_dims(confs, -1)], 3)
    veYid =  salph * weight_veY


    self.fetch += [_probs, confs, conid, cooid, proid, disid, _dista]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs, _dista], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid, disid], 3)


    if self.FLAGS.alpha: #alphaを使う場合
         '''
         #adjusted_alpha      = tf.sqrt(tf.exp(   alpha[:,:,:,:1]) * anchors[:,:,:,3:4] / np.reshape([W], [1, 1, 1, 1]))
         #adjusted_net_out = tf.concat([adjusted_net_out, adjusted_alpha], 3)
         #adjusted_vecX      = tf.sqrt(tf.exp(   vecX[:,:,:,:1]) * anchors[:,:,:,3:4] / np.reshape([W], [1, 1, 1, 1]))
		 #adjusted_vecY      = tf.sqrt(tf.exp(   vecY[:,:,:,:1]) * anchors[:,:,:,4:5] / np.reshape([W], [1, 1, 1, 1]))

         #import pdb; pdb.set_trace()
         anchor_vec          = anchors[:,:,:,3:5] / np.reshape([W,H], [1, 1, 1, 2])
         #anchor_            = anchors[:,:,:,3:4] / np.reshape([W], [1, 1, 1, 1]) * np.reshape([math.pi], [1, 1, 1, 1]) #ベクトル用のアンカーを２次元分用意
         #anchor_vec_x       = np.cos(anchor_)
         #anchor_vec_y       = np.sin(anchor_)
         #anchor_vec         = np.concatenate([anchor_vec_x, anchor_vec_y], 3)
         adjusted_vec       = tf.concat( [vecX, vecY], 3)#出力のうちのベクトルを用意
         adjusted_vec       = tf.add( adjusted_vec[:,:,:,:], anchor_vec) #残差を学習するようにアンカーと足す
         #trueを整理
         _vec               = tf.concat([_vecX, _vecY], 3)#真の値の分を用意
         _vec_abs           = tf.norm(_vec,axis=3)#真の値のベクトルの絶対値を
         _vec_abs           = tf.reshape(_vec_abs,[-1, H*W, B, 1])#型を整える
         adjusted_vec_abs   = tf.norm(adjusted_vec,axis=3)#推定値にも同じように
         adjusted_vec_abs   = tf.reshape(adjusted_vec_abs,[-1, H*W, B, 1])#
         vec_dot            = tf.matmul(adjusted_vec , _vec, transpose_b=True)#内積を計算するので要素ごとに掛け算して
         #vec_dot            = vec_dot[:,:,:,:1]
         vec_dot            = vec_dot * sasaki #10x10で用意しておいた単位行列を掛け合わせて対角成分のみつかう
         vec_dot            = tf.reduce_sum(vec_dot,axis = 3)#そのようをを３次元目で合算して内積を出す
         vec_dot            = tf.reshape(vec_dot,[-1, H*W, B, 1])#それを型があうようにする¥
         vec_abs_fin        = tf.multiply(adjusted_vec_abs,_vec_abs)
         vec_abs_fin        = tf.add(vec_abs_fin, 0.001)

         difal              = tf.subtract(1., tf.divide(vec_dot, vec_abs_fin))  #1から内積/絶対値の積を引いたもの
         #difal               = tf.abs(vec_dot)
         #adjusted_net_out = tf.concat([adjusted_net_out, adjusted_vecX, adjusted_vecY], 3)


         #self.fetch += [alpid, _alpha]
         self.fetch += [veXid, _vec,]
         #true = tf.concat([true, _alpha], 3)
         #wght = tf.concat([wght, alpid], 3)
         #true = tf.concat([true, _vecX, _vecY], 3)
         wght = tf.concat([wght, veXid], 3)
         '''

    print('Building {} loss'.format(m['model']))
    #loss = tf.pow(tf.multiply(_kayu , adjusted_net_out - true), 2)
    loss = tf.pow(adjusted_net_out - true, 2)
    #loss = tf.concat([loss,difal], 3)
    loss = tf.multiply(loss, wght)
    if self.FLAGS.alpha:
        loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + 1 +1 + C)])
    else:
        loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)



def hoge(sassa):
    hiro = sassa * 99

    return hiro
