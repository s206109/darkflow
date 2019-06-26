from darkflow.net.build import TFNet
import cv2
import numpy as np
import os
import glob
import json
import re
import pdb
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from darkflow.utils import box
from darkflow.utils.pascal_voc_clean_xml_evaluation import pascal_voc_clean_xml
from darkflow.utils import process
import math


#-----------------------------
# parameters
visualPath = 'visualization'

labels = ['car','negative']
threshold = 0.7
#-----------------------------

#-----------------------------
# load config values
#_, meta = process.parser('cfg/tiny-yolo-kitti-3d.cfg')
#_, meta = process.parser('cfg/tiny-yolo-kitti-3d_13x13.cfg')
_, meta = process.parser('cfg/tiny-yolo-kitti-3d-10.cfg')
#-----------------------------

#-----------------------------
# load gt annotations from xml files as gtBoxes
print('extract annotations data')
gtBoxes = pascal_voc_clean_xml(meta['annotation_path'], labels, exclusive = False) #ここでようやくデータセット読み込み

# sort to make the correspondence between gtBoxes and predBoxes
gtBoxes.sort()
#-----------------------------


#-----------------------------
# load predicted boxes as predBoxes
# jsonの読み込み
cur_dir = os.getcwd()
os.chdir(meta['json_path'])
jsonFiles = glob.glob('*.json')

predBoxes = [0 for re2 in range(len(jsonFiles))] #jsonfileの数だけ要素を儲ける
for i, file in enumerate(jsonFiles):
    with open(file) as f:
       js = json.load(f)
       jnum = len(js)
       cdBox = [[0 for ii in range(7)] for iii in range(jnum)]
       for j in range(jnum):
           #import pdb; pdb.set_trace()
           cdBox[j][0] = js[j]["label"]
           cdBox[j][6] = js[j]["confidence"]
           cdBox[j][1] = js[j]["topleft"]["x"]
           cdBox[j][2] = js[j]["topleft"]["y"]
           cdBox[j][3] = js[j]["bottomright"]["x"]
           cdBox[j][4] = js[j]["bottomright"]["y"]
           cdBox[j][5] = js[j]["dist"]
       cdBox.insert(0,int(re.sub(r'\D', '',file))) # ファイル名からどのファイルかインデックスとして抽出
    predBoxes[i] = cdBox

predBoxes.sort()
os.chdir(cur_dir)
#-----------------------------
bugid = [ 694,  823, 1134, 2029, 2368, 2826, 3348, 3629, 3649]
bugname = []
#-----------------------------
# for each image, compute IoU between predBox and gtBox
# and select the gtBox with the highest IoU

# dataframe for result records
import pdb; pdb.set_trace()
resultDF = pd.DataFrame(columns = ['iou','pc','px','py','pw','ph','pz','gc','gx','gy','gw','gh','gz','ga','pz-gz','garea','fn'])
for dInd in np.arange(0,len(predBoxes)): #dInd = 何ファイル目なのかの数
    #print("======")
    for pInd in np.arange(1,len(predBoxes[dInd])): #1つ目はファイル名なので。物体の数だけまわす
        predBox = box.BoundBox(2)
        predBox.c = predBoxes[dInd][pInd][0]
        predBox.x = predBoxes[dInd][pInd][1]
        predBox.y = predBoxes[dInd][pInd][2]
        predBox.w = predBoxes[dInd][pInd][3] - predBoxes[dInd][pInd][1]
        predBox.h = predBoxes[dInd][pInd][4] - predBoxes[dInd][pInd][2]
        predBox.z = predBoxes[dInd][pInd][5]
        predBox.filenum = predBoxes[dInd][0] #filename取得
        ious = []
        gtBox = [box.BoundBox(2) for i in np.arange(1,len(gtBoxes[dInd]))] #物体の数だけgt入れる箱を作る

        for gInd in np.arange(1,len(gtBoxes[dInd])):
            if predBox.c != gtBoxes[dInd][gInd][0]:
                 ious.append(0.0) #classが違えば　iouをアペンドしてから飛ばす
                 continue #classが違えば飛ばす

            gtBox[gInd-1].c = gtBoxes[dInd][gInd][0]
            gtBox[gInd-1].x = gtBoxes[dInd][gInd][1]
            gtBox[gInd-1].y = gtBoxes[dInd][gInd][2]
            gtBox[gInd-1].w = gtBoxes[dInd][gInd][3] - gtBoxes[dInd][gInd][1]
            gtBox[gInd-1].h = gtBoxes[dInd][gInd][4] - gtBoxes[dInd][gInd][2]
            gtBox[gInd-1].z = gtBoxes[dInd][gInd][5]
            gtBox[gInd-1].alpha = gtBoxes[dInd][gInd][6]

            ious.append(box.box_iou(predBox, gtBox[gInd-1]))

        if len(ious) == 0: continue

        ious = np.array(ious)
        maxInd = np.argmax(ious) #iouが最大になっているインデックスを返す

        resultDF = resultDF.append(pd.Series([np.max(ious),
                           predBox.c, predBox.x, predBox.y, predBox.w, predBox.h, predBox.z,
                           gtBox[maxInd].c, gtBox[maxInd].x, gtBox[maxInd].y, gtBox[maxInd].w, gtBox[maxInd].h, gtBox[maxInd].z,gtBox[maxInd].alpha,(predBox.z   - gtBox[maxInd].z) ,gtBox[maxInd].w * gtBox[maxInd].h, predBox.filenum],
                           index=resultDF.columns),ignore_index=True)

#-----------------------------
#TEST
#import pdb; pdb.set_trace()
surveyInd  = np.where((resultDF['iou'] > 0.7) & (resultDF['gh'] > 25))[0]

surveyx = resultDF.ix[surveyInd]['ga']
#surveyx2 = resultDF.ix[surveyInd2]['p_area-g_area']

surveyy = resultDF.ix[surveyInd]['gw']
#surveyy2 = resultDF.ix[surveyInd2]['iou']

surveyy_g = resultDF.ix[surveyInd]['gz']
surveyx_g = resultDF.ix[surveyInd]['ga']
import pdb; pdb.set_trace()
mejirushi  = np.zeros(10)
mejirushiy = np.zeros(10)
widthRatio = 1242/13
heightRatio = 370/13
#for inda in range(10):
#    mejirushi[inda]  = widthRatio  * meta['anchors'][3*inda]
#    mejirushiy[inda] = heightRatio * meta['anchors'][3*inda + 1]


plt.scatter(surveyx, surveyy,   c='b', s = 5,label = None)
#plt.scatter(mejirushi, mejirushiy,   c='r', s = 15,label = None)
#plt.scatter(surveyx2, surveyy2,   c='r', s = 5,label = None)
#for ssk in mejirushi:
     #plt.vlines([ssk], -15, 15, "black", linestyles='dashed')
#plt.scatter(surveyx_g, surveyy_g,   c='r', label = 'test_data')

# 凡例を表示する
plt.legend()
plt.xlabel('GT object width [pixel]',fontsize = 20)
plt.ylabel('GT object height [pixel] ',fontsize = 20)

# グラフのタイトルを設定する
plt.title("Distribution of height and width",fontsize = 20)
plt.savefig(os.path.join(visualPath,'cont_new.png'))
# 表示する
plt.show()
#-----------------------------

"""
#-----------------------------
#TEST
import pdb; pdb.set_trace()
surveyInd = np.where(resultDF['iou'] > 0.7)[0] #iou0.7のものを用意
surveyx = resultDF.ix[surveyInd]['ga']
surveyy = resultDF.ix[surveyInd]['pz-gz']
surveyy_g = resultDF.ix[surveyInd]['gz']
surveyx_g = resultDF.ix[surveyInd]['ga']
mejirushiy = [0, 0, 0, 0, 0]
mejirushi = [-1*math.pi,(-1*math.pi)/2, 0 ,math.pi/2,math.pi]
plt.scatter(surveyx, surveyy,   c='b', s = 5,label = None)
for ssk in mejirushi:
     plt.vlines([ssk], -15, 15, "black", linestyles='dashed')
#plt.scatter(surveyx_g, surveyy_g,   c='r', label = 'test_data')

# 凡例を表示する
plt.legend()
plt.xlabel('object orientation α [rad]',fontsize = 18)
plt.ylabel('distance error',fontsize = 18)

# グラフのタイトルを設定する
plt.title("Distribution of distance error",fontsize = 18)
plt.savefig(os.path.join(visualPath,'cont_test.png'))
# 表示する
plt.show()
#-----------------------------

#-----------------------------

# compute error
print(bugname)
#import pdb; pdb.set_trace()
inds = np.where((resultDF['iou'] > 0.7) & (resultDF['gz'] <= 10) & (resultDF['gh'] > 25))[0]
error10 = np.mean(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))
std10 = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['gz'] > 10) & (resultDF['gz'] <= 20) & (resultDF['gh'] > 25))[0]
error20 = np.mean(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))
std20 = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['gz'] > 20) & (resultDF['gz'] <= 30) & (resultDF['gh'] > 25))[0]
error30 = np.mean(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))
std30 = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['gz'] > 30) & (resultDF['gz'] <= 40) & (resultDF['gh'] > 25))[0]
error40 = np.mean(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))
std40 = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['gz'] > 40) & (resultDF['gh'] > 25))[0]
error40over = np.mean(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))
std40over = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

import pdb; pdb.set_trace()
#-----------------------------

#-----------------------------
# plot distance prediction error
plt.plot(['10','20','30','40','40 over'],[error10, error20, error30, error40, error40over])
plt.plot(['10','20','30','40','40 over'],[1.5,1,1.85,2.3,3])
plt.xlabel('true distance')
plt.ylabel('absolute error')
plt.savefig(os.path.join(visualPath,'error_test.png'))
plt.show()
#-----------------------------
pdb.set_trace()



#img = cv2.imread('data/kitti/set1/PNGImagesTest/000002.png')
img = cv2.imread('test.jpg')
# 解析を行う
items = tfnet.return_predict(img)
# 検出できたものを確認
import pdb; pdb.set_trace()
print(items)

for item in items:
    # 四角を描くのに必要な情報とラベルを取り出す
    tlx = item['topleft']['x']
    tly = item['topleft']['y']
    brx = item['bottomright']['x']
    bry = item['bottomright']['y']
    label = item['label']
    conf = item['confidence']
    dist = item['distance']
    print(item)
    # 自信のあるものを表示
    if conf > 0.05:

        for i in labels:
            if label == i:
                class_num = labels.index(i)
                break

"""
