from darkflow.net.build import TFNet
import cv2
import numpy as np
import os
import glob
import json
import re
import pdb
import pandas as pd
import matplotlib.pylab as plt
from darkflow.utils import box
from darkflow.utils.pascal_voc_clean_xml_evaluation import pascal_voc_clean_xml
from darkflow.utils import process
import math


"""
def MUKI(arg):
    if   arg <     math.pi/4 and arg >= -1*math.pi/4:
        muki = "right"
    elif arg <=  3*math.pi/4 and arg >     math.pi/4:
        muki = "back"
    elif abs(arg) > 3*math.pi/4 or arg == 3*math.pi/4:
        muki = "left"
    elif arg >= -3*math.pi/4 and arg < -1*math.pi/4:
        muki = "front"



    return muki
"""
#-----------------------------
# parameters
visualPath = 'visualization'

labels = ['car','Truck']
threshold = 0.7
#-----------------------------

#-----------------------------
# load config values
_, meta = process.parser('cfg/tiny-yolo-kitti-3d.cfg')
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
os.chdir('data/kitti/set1/PNGImagesTest/out_53000_final')
jsonFiles = glob.glob('*.json')

predBoxes = [0 for re2 in range(len(jsonFiles))]
for i, file in enumerate(jsonFiles):
    with open(file) as f:
       js = json.load(f)
       jnum = len(js)
       cdBox = [[0 for ii in range(9)] for iii in range(jnum)]
       for j in range(jnum):
           #import pdb; pdb.set_trace()
           cdBox[j][0] = js[j]["label"]
           cdBox[j][6] = js[j]["confidence"]
           cdBox[j][1] = js[j]["topleft"]["x"]
           cdBox[j][2] = js[j]["topleft"]["y"]
           cdBox[j][3] = js[j]["bottomright"]["x"]
           cdBox[j][4] = js[j]["bottomright"]["y"]
           cdBox[j][5] = js[j]["dist"]
           #cdBox[j][7] = js[j]["alph"]
           #cdBox[j][7] = js[j]["vecX"]
           #cdBox[j][8] = js[j]["vecY"]
       cdBox.insert(0,int(re.sub(r'\D', '',file))) # ファイル名からどのファイルかインデックスとして抽出
    predBoxes[i] = cdBox

predBoxes.sort()
#import pdb; pdb.set_trace()
os.chdir(cur_dir)
#-----------------------------
subarasii = [   15,    49,   334,   383,   788,  2181,  2466,  2846,  2948,
        3473,  4325,  4343,  5151,  5645,  6325,  6913,  7313,  8005,
        8053,  8531,  8667,  9131,  9903, 10169]
bugid = [    1,    75,   738,  1755,  1829,  1839,  2036,  2130,  2554,3529,  3943,  4454,  4826,  4875,  4888,  4926,  5302,  5428, 5532,  5652,  7285,  7633,  7953,  8349,  8419,  8546,  8882, 9218,  9305,  9505, 10015, 10258, 10472]
bugname = []
#-----------------------------
# for each image, compute IoU between predBox and gtBox
# and select the gtBox with the highest IoU

# dataframe for result records
resultDF = pd.DataFrame(columns = ['iou','pc','px','py','pw','ph','pz','gc','gx','gy','gw','gh','gz','ad','pz-gz','ga','fn'])
for dInd in np.arange(0,len(predBoxes)): #dInd = 何ファイル目なのかの数
    for pInd in np.arange(1,len(predBoxes[dInd])): #1つ目はファイル名なので。物体の数だけまわす
        predBox = box.BoundBox(2)
        predBox.c = predBoxes[dInd][pInd][0]
        predBox.x = predBoxes[dInd][pInd][1]
        predBox.y = predBoxes[dInd][pInd][2]
        predBox.w = predBoxes[dInd][pInd][3] - predBoxes[dInd][pInd][1]
        predBox.h = predBoxes[dInd][pInd][4] - predBoxes[dInd][pInd][2]
        predBox.z = predBoxes[dInd][pInd][5]
        predBox.alpha = predBoxes[dInd][pInd][7]
        #predBox.vecX = predBoxes[dInd][pInd][7]
        #if predBoxes[dInd][pInd][7] > 0.5:import pdb; pdb.set_trace()
        #predBox.vecY = predBoxes[dInd][pInd][8]
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
            #gtBox[gInd-1].vecX = gtBoxes[dInd][gInd][6]
            #gtBox[gInd-1].vecY = gtBoxes[dInd][gInd][7]
            gtBox[gInd-1].alpha = gtBoxes[dInd][gInd][8]
            ious.append(box.box_iou(predBox, gtBox[gInd-1]))

        if len(ious) == 0: continue

        ious = np.array(ious)
        maxInd = np.argmax(ious) #iouが最大になっているインデックスを返す
        #vecX_pr= 2*(predBox.vecX)-1
        #vecY_pr= 2*(predBox.vecY)-1
        #alp_pr = math.atan2( vecY_pr, vecX_pr  )
        #alp_gt = math.atan2(gtBox[maxInd].vecY,gtBox[maxInd].vecX)
        alphadif = predBox.alpha - gtBox[maxInd].alpha
        #vecXdif = vecX_pr - gtBox[maxInd].vecX
        #vecYdif = vecY_pr - gtBox[maxInd].vecY
        #vecXdif = vecX_pr
        #vecYdif = vecY_pr
        #if alphadif > math.pi:

        #    alphadif =  2 * math.pi - alphadif
        #elif alphadif < -1 * math.pi:
        #    alphadif = -2 * math.pi - alphadif
        resultDF = resultDF.append(pd.Series([np.max(ious),
                           predBox.c, predBox.x, predBox.y, predBox.w, predBox.h, predBox.z,
                           gtBox[maxInd].c, gtBox[maxInd].x, gtBox[maxInd].y, gtBox[maxInd].w, gtBox[maxInd].h, gtBox[maxInd].z,
                           alphadif,
                           (predBox.z   - gtBox[maxInd].z) ,gtBox[maxInd].alpha, predBox.filenum],
                           index=resultDF.columns),ignore_index=True)



"""
#-----------------------------
#ALPHATEST
import pdb; pdb.set_trace()
surveyInd = np.where(resultDF['iou'] > 0.7)[0] #iou0.7のものを用意
surveyx = resultDF.ix[surveyInd]['ga']
surveyy = resultDF.ix[surveyInd]['ad']
#surveyy_g = resultDF.ix[surveyInd]['gz']
#surveyx_g = resultDF.ix[surveyInd]['ga']
mejirushiy = [0, 0, 0, 0, 0]
mejirushi = [-1*math.pi,(-1*math.pi)/2, 0 ,math.pi/2,math.pi]
plt.scatter(surveyx, surveyy,   c='b', s = 5, label = 'test_data')
#for ssk in mejirushi:
#     plt.vlines([ssk], -15, 15, "red", linestyles='dashed')
#plt.scatter(surveyx_g, surveyy_g,   c='r', label = 'test_data')

# 凡例を表示する
plt.legend()

# グラフのタイトルを設定する
plt.title("test_datas")

# 表示する
plt.show()
#-----------------------------


import pdb; pdb.set_trace()

#-----------------------------
#TEST

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

# 表示する
plt.show()
#-----------------------------

"""

# compute error
import pdb; pdb.set_trace()
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



#-----------------------------
"""

pdb.set_trace()


#-----------------------------

# alpha ga umakudekiteiru mono

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['gz'] <= 10) & (resultDF['gh'] > 25))[0]
error10_a = np.mean(np.abs((resultDF.ix[inds].ad).values))
std10_a = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['gz'] > 10) & (resultDF['gz'] <= 20) & (resultDF['gh'] > 25))[0]
error20_a = np.mean(np.abs((resultDF.ix[inds].ad).values))
std20_a = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['gz'] > 20) & (resultDF['gz'] <= 30) & (resultDF['gh'] > 25))[0]
error30_a = np.mean(np.abs((resultDF.ix[inds].ad).values))
std30_a = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['gz'] > 30) & (resultDF['gz'] <= 40) & (resultDF['gh'] > 25))[0]
error40_a = np.mean(np.abs((resultDF.ix[inds].ad).values))
std40_a = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['gz'] > 40) & (resultDF['gh'] > 25))[0]
error40over_a = np.mean(np.abs((resultDF.ix[inds].ad).values))
std40over_a = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

#-----------------------------
#-----------------------------

"""
#-----------------------------
# plot distance prediction error
plt.plot(['[0-10]','[10-20]','[20-30]','[30-40]','[40 over]'],[error10, error20, error30, error40, error40over],label = 'estimation with 2.5D + orientation anchor',color = 'blue')
#plt.plot(['[0-10]','[10-20]','[20-30]','[30-40]','[40 over]'],[error10_a, error20_a, error30_a, error40_a, error40over_a],label = 'orientation with 2.5D + orientation anchor')
#plt.plot(['1:[-3pi/4<]','2:[-3pi/4:-pi/4]','3:[-pi/4:pi/4]','4:pi/4:3pi/4]','5:[<3pi/4]'],[error1, error2, error3, error4, error5],label = 'estimation with 2.5D + orientation anchor')
#plt.plot(['1:[-3pi/4<]','2:[-3pi/4:-pi/4]','3:[-pi/4:pi/4]','4:pi/4:3pi/4]','5:[<3pi/4]'],[bunsan1, bunsan2, bunsan3, bunsan4, bunsan5],label = 'bunsan with 2.5D + orientation anchor')
#plt.plot(['10','20','30','40','40 over'],[1.5,1,1.85,2.3,3])
#plt.plot(['10','20','30','40','40 over'],[1.3878909524222403, 1.7428688349630319, 2.771728648535813, 3.5718634061115546, 3.5744018749480553])


plt.plot(['[0-10]','[10-20]','[20-30]','[30-40]','[40 over]'],[1.1577788484456362, 1.6437140840763669, 3.6163843362981618, 5.0210139905144189, 3.9699841791788741],label = 'estimation with 2.5D anchor',color = 'red')

#plt.xlabel('true distance',fontsize = 18)
std_error = [std10/10,std20/10 ,std30/10, std40/10, std40over/10]
std_error2 = [0.088569044128493188, 0.1430281019268857, 0.29578555799245304, 0.36913817919553921, 0.33677865394882478]

plt.errorbar(['[0-10]','[10-20]','[20-30]','[30-40]','[40 over]'], [error10, error20, error30, error40, error40over], std_error,fmt = 'o',color = 'blue')
plt.errorbar(['[0-10]','[10-20]','[20-30]','[30-40]','[40 over]'], [1.1577788484456362, 1.6437140840763669, 3.6163843362981618, 5.0210139905144189, 3.9699841791788741], std_error2,fmt = 'o',color = 'red')
plt.legend(fontsize = 10)
plt.xlabel('true distance[m]',fontsize = 16)
#plt.ylabel('absolute error',fontsize = 18)
plt.ylabel('absolute error',fontsize = 16)
plt.savefig(os.path.join(visualPath,'true_distance_vs_estimation_absolute_errror.png'))
plt.show()

"""
# compute error
import pdb; pdb.set_trace()
inds = np.where((resultDF['iou'] > 0.7) & (resultDF['ga'] <=  -3 * math.pi/ 4) & (resultDF['gh'] > 25))[0]
error1 = np.mean(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))
#ave1   = np.mean((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values)
#bunsan1 = np.mean(((resultDF.ix[inds].pz - resultDF.ix[inds].pz).values - ave1)*((resultDF.ix[inds].pz - resultDF.ix[inds].pz).values - ave1))
bunsan1 = np.var((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values)
std10 = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['ga']  >  -3 * math.pi/ 4) & (resultDF['ga'] <= -math.pi/ 4) & (resultDF['gh'] > 25))[0]
error2 = np.mean(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))
#ave2   = np.mean((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values)
#bunsan2 = np.mean(((resultDF.ix[inds].pz - resultDF.ix[inds].pz).values - ave2)*((resultDF.ix[inds].pz - resultDF.ix[inds].pz).values - ave2))
bunsan2 = np.var((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values)
std20 = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['ga']  >     -math.pi / 4) & (resultDF['ga'] <=  math.pi/ 4) & (resultDF['gh'] > 25))[0]
error3 = np.mean(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))
#ave3   = np.mean((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values)
#bunsan3 = np.mean(((resultDF.ix[inds].pz - resultDF.ix[inds].pz).values - ave3)*((resultDF.ix[inds].pz - resultDF.ix[inds].pz).values - ave3))
bunsan3 = np.var((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values)
std30 = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['ga'] >       math.pi / 4) & (resultDF['ga'] <= 3 * math.pi/ 4) & (resultDF['gh'] > 25))[0]
error4 = np.mean(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))
#ave4   = np.mean((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values)
#bunsan4 = np.mean(((resultDF.ix[inds].pz - resultDF.ix[inds].pz).values - ave4)*((resultDF.ix[inds].pz - resultDF.ix[inds].pz).values - ave4))
bunsan4 = np.var((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values)
std40 = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))

inds = np.where((resultDF['iou'] > 0.7) & (resultDF['ga'] > 3 * math.pi/ 4) & (resultDF['gh'] > 25))[0]
error5 = np.mean(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))
#ave5   = np.mean((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values)
#bunsan5 = np.mean(((resultDF.ix[inds].pz - resultDF.ix[inds].pz).values - ave5)*((resultDF.ix[inds].pz - resultDF.ix[inds].pz).values - ave5))
bunsan5 = np.var((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values)
std40over = np.std(np.abs((resultDF.ix[inds].gz - resultDF.ix[inds].pz).values))





#-----------------------------
# plot distance prediction error
#plt.plot(['[0-10]','[10-20]','[20-30]','[30-40]','[40 over]'],[error10, error20, error30, error40, error40over],label = 'estimation with 2.5D + orientation anchor')
plt.plot(['1:[-3pi/4<]','2:[-3pi/4:-pi/4]','3:[-pi/4:pi/4]','4:pi/4:3pi/4]','5:[<3pi/4]'],[error1, error2, error3, error4, error5],label = 'estimation with 2.5D + orientation anchor')
plt.plot(['1:[-3pi/4<]','2:[-3pi/4:-pi/4]','3:[-pi/4:pi/4]','4:pi/4:3pi/4]','5:[<3pi/4]'],[bunsan1, bunsan2, bunsan3, bunsan4, bunsan5],label = 'bunsan with 2.5D + orientation anchor')
#plt.plot(['10','20','30','40','40 over'],[1.5,1,1.85,2.3,3])
#plt.plot(['10','20','30','40','40 over'],[1.3878909524222403, 1.7428688349630319, 2.771728648535813, 3.5718634061115546, 3.5744018749480553])


#plt.plot(['[0-10]','[10-20]','[20-30]','[30-40]','[40 over]'],[1.1577788484456362, 1.6437140840763669, 3.6163843362981618, 5.0210139905144189, 3.9699841791788741],label = 'estimation with 2.5D anchor')
plt.legend(fontsize = 10)
#plt.xlabel('true distance',fontsize = 18)
plt.xlabel('true orientation',fontsize = 18)
#plt.ylabel('absolute error',fontsize = 18)
plt.ylabel('absolute error',fontsize = 18)
plt.savefig(os.path.join(visualPath,'true_distance_vs_estimation_absolute_errror.png'))
plt.show()
#-----------------------------

pdb.set_trace()
"""
