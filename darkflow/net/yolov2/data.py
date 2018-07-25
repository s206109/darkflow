from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from ..yolo.predict import preprocess
from ..yolo.data import shuffle
from copy import deepcopy
import pickle
import numpy as np
import os

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
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)#for文用に同じものを複製
    #import pdb; pdb.set_trace()
    path = os.path.join(self.FLAGS.dataset, jpg)
    img = self.preprocess(path, allobj)#ここで入力を

    # Calculate regression target
    cellx = 1. * w / W #画像の横幅を１グリッドあたりのピクセル数
    celly = 1. * h / H #画像の縦幅１グリッドあたりのピクセル数
    #import pdb; pdb.set_trace()
    #
    for obj in allobj:
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax 物体の中心座標
        centery = .5*(obj[2]+obj[4]) #ymin, ymax 物体の中心座標
        cx = centerx / cellx #どこのセルにあるかの番号
        cy = centery / celly #どこのセルにあるかの番号
        #import pdb; pdb.set_trace()
        if cx >= W or cy >= H: return None, None #１３以上なら画面外になってしまうから
        obj[3] = float(obj[3]-obj[1]) / w #画像あたりのBBの横幅比率
        obj[4] = float(obj[4]-obj[2]) / h #画像あたりのBBの縦幅比率
        obj[3] = np.sqrt(obj[3]) #　そのルート
        obj[4] = np.sqrt(obj[4]) #　そのルート
        obj[1] = cx - np.floor(cx) # centerx　この値が０でなければ次の番号のセルであるということ
        obj[2] = cy - np.floor(cy) # centery　この値が０でなければ次の番号のセルであるということ
        import pdb; pdb.set_trace()
        obj += [int(np.floor(cy) * W + np.floor(cx))]#この数字はなんのために使うのか謎。７番目の値

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([H*W,B,C])
    confs = np.zeros([H*W,B])
    coord = np.zeros([H*W,B,4])
    proid = np.zeros([H*W,B,C])
    prear = np.zeros([H*W,4])
    #import pdb; pdb.set_trace()
    for obj in allobj:
        probs[obj[6], :, :] = [[0.]*C] * B
        probs[obj[6], :, labels.index(obj[0])] = 1.
        proid[obj[6], :, :] = [[1.]*C] * B
        coord[obj[6], :, :] = [obj[1:5]] * B
        prear[obj[6],0] = obj[1] - obj[3]**2 * .5 * W # xleft
        prear[obj[6],1] = obj[2] - obj[4]**2 * .5 * H # yup
        prear[obj[6],2] = obj[1] + obj[3]**2 * .5 * W # xright
        prear[obj[6],3] = obj[2] + obj[4]**2 * .5 * H # ybot
        confs[obj[6], :] = [1.] * B

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft;
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft,
        'botright': botright
    }
    #import pdb; pdb.set_trace()
    return inp_feed_val, loss_feed_val
