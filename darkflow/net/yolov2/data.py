from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from ..yolo.predict import preprocess
from ..yolo.data import shuffle
from copy import deepcopy
import pickle
import numpy as np
import os
import pdb
import math

def _batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's
    input & loss layer correspond to this chunk
    """
    meta = self.meta
    labels = meta['labels']

    H, W, _ = meta['out_size']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']

    # preprocess
    maxz = meta['maxz'] # 距離の最大値を仮設定
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)#for文用に同じものを複製
    path = os.path.join(self.FLAGS.dataset, jpg)
    img = self.preprocess(path, allobj)#ここで入力を
    #import pdb; pdb.set_trace()
    # Calculate regression target
    cellx = 1. * w / W #画像の横幅を１グリッドあたりのピクセル数
    celly = 1. * h / H #画像の縦幅１グリッドあたりのピクセル数
    #
    for obj in allobj: #[class, xmin, ymin, xmax, ymax, z]
        #if obj[0] == "Truck": continue
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax 物体の中心座標
        centery = .5*(obj[2]+obj[4]) #ymin, ymax 物体の中心座標
        cx = centerx / cellx #どこのセルにあるかの番号
        cy = centery / celly #どこのセルにあるかの番号

        if cx >= W or cy >= H:
           return None, None #１３以上なら画面外になってしまうから

        obj[3] = float(obj[3]-obj[1]) / w #画像あたりのBBの横幅比率
        obj[4] = float(obj[4]-obj[2]) / h #画像あたりのBBの縦幅比率
        obj[5] = obj[5] / maxz #最大距離に対する距離の比率
        if obj[5] < 0: obj[5] = 0
        #if obj[6] < 0:obj[6] += math.pi
        #obj[6] = abs(math.cos(obj[6]))
        #if obj[6] >  math.pi:
        #    obj[6] = math.pi
        #if obj[6] < -math.pi:
        #    obj[6] = -math.pi
        #obj[6] = obj[6]
        obj[3] = np.sqrt(obj[3]) #　画像に対するBBの横幅比のルート
        obj[4] = np.sqrt(obj[4]) #　画像に対するBBの縦幅比のルート
        obj[5] = np.sqrt(obj[5]) #　距離の比率のルート
        obj[1] = cx - np.floor(cx) # セル番号と少数切り捨てのセル番号の差
        obj[2] = cy - np.floor(cy) # セル番号と少数切り捨てのセル番号の差
        obj += [int(np.floor(cy) * W + np.floor(cx))]#左上からラスタースキャンで数えて、BBが属するセル番号(距離にも応用可能？）
    # show(im, allobj, S, w, h, cellx, celly) # unit test
    #import pdb; pdb.set_trace()
    # Calculate placeholders' values
    # 値を入れるために特定の和の要素の配列を確保
    #probs = np.zeros([H*W,B,C]) #169x5x2セルごとの各クラスへの所属確率
    probs = np.zeros([H*W,B,C]) #169x5x2セルごとの各クラスへの所属確率
    probs[:, :, 1] = 1.         #一旦全セルに関して、背景クラスを１に
    confs = np.zeros([H*W,B]) #169x5 セルごとの各BBの信頼度
    coord = np.zeros([H*W,B,4]) #169x5x4  セルごとのBBの座標
    proid = np.zeros([H*W,B,C]) #169x5x2
    prear = np.zeros([H*W,4]) #169x4
    dista = np.zeros([H*W,B,1])#169x5x1 セルごとの各BBの物体との距離
    #vecX  = np.zeros([H*W,B,1])
    #vecY  = np.zeros([H*W,B,1])
    #alpha = np.zeros([H*W,B,1])#169x5x1 セルごとの各BBの物体の角度
    #import pdb; pdb.set_trace()
    for obj in allobj: #[classs, xセル番号, yセル番号, w幅比率のルート,　h幅比率のルート, 距離比率のルート, セル番号]
        #if obj[0] == "Truck": continue
        import pdb; pdb.set_trace()
        #obj[7] = ラスタースキャンで数えたセル番号
        probs[obj[6], :, :] = [[0.]*C] * B #169セルのうちの物体があるセルにクラスの数だけ要素を設けている
        #############################################################################################
        #分類タスクの場合
        #probs[obj[6], :, labels.index(obj[0])] = 1.   #そのうち正解物体の方の確率を１とする
        #proid[obj[6], :, :] = [[1.]*C] * B #物体があるセルのクラスにかかわらず１を代入
        #############################################################################################
        #検出タスクの場合（物体か背景かの分類のため）
        probs[obj[6], :, labels.index(obj[0])] = 1.   #そのうち正解物体のあるセルの方の確率を１とする
        probs[obj[6], :, 1] = 0.   #忘れずに正解物体の方の背景クラスを０とする
        #############################################################################################
        coord[obj[6], :, :] = [obj[1:5]] * B # x,y,w,hのルート比率を、アンカーの数だけそれぞれに同じものを代入
        prear[obj[6],0] = obj[1] - obj[3]**2 * .5 * W # xleft座標
        prear[obj[6],1] = obj[2] - obj[4]**2 * .5 * H # yup座標
        prear[obj[6],2] = obj[1] + obj[3]**2 * .5 * W # xright座標
        prear[obj[6],3] = obj[2] + obj[4]**2 * .5 * H # ybot座標
        confs[obj[6], :] = [1.] * B #物体が存在するセルのBBをすべて信頼度を１とする
        dista[obj[6], :, :] = [[obj[5]]] * B # 距離の比率をアンカーの数だけそれぞれに同じものを代入
        #vecX[obj[7], :, :] = [[(math.cos(obj[6])+1)/2]] * B # cosαをアンカーの数だけそれぞれに同じものを代入
        #vecX[obj[7], :, :] = [[math.cos(obj[6])]] * B # cosαをアンカーの数だけそれぞれに同じものを代入
        #vecY[obj[7], :, :] = [[(math.sin(obj[6])+1)/2]] * B # sinαをアンカーの数だけそれぞれに同じものを代入
        #vecY[obj[7], :, :] = [[math.sin(obj[6])]] * B # sinαをアンカーの数だけそれぞれに同じものを代入
    #import pdb; pdb.set_trace()
    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1) #単純にBBの左上の座標
    botright = np.expand_dims(prear[:,2:4], 1) #単純にBBの右下の座標
    wh = botright - upleft; #BBの縦横の幅
    #import pdb; pdb.set_trace()
    area = wh[:,:,0] * wh[:,:,1] #セルに物体があった場合のBBの面積
    upleft   = np.concatenate([upleft] * B, 1) #これをBBの数（５）分だけ用意する
    botright = np.concatenate([botright] * B, 1)#これをBBの数（５）分だけ用意する
    areas = np.concatenate([area] * B, 1 )#これをBBの数（５）分だけ用意する
    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer
    #import pdb; pdb.set_trace()
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft,
        'botright': botright, 'dista':dista
    }

    return inp_feed_val, loss_feed_val
