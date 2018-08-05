from darkflow.net.build import TFNet
import cv2
import numpy as np
import os
import glob
import json
import re
from darkflow.utils.pascal_voc_clean_xml_ex import pascal_voc_clean_xml



#options = {"model": "cfg/yolo.cfg" ,"load":"bin/yolo.weights", "threshold": 0.1,  "json": True, "imgdir": "data/kitti/set1/PNGImagesTest"}
options = {"model": "cfg/tiny-yolo-kitti-3d.cfg" ,"load":33000, "threshold": 0.1, "json": True}
tfnet = TFNet(options)

#アノテーションの読み込み

meta = tfnet.meta #おそらくcfgから取ってきた　cfgの設定値
print("--------")
print(meta)
print("--------")
#ann = self.FLAGS.annotation #

print('extract annotations data')
dumps = pascal_voc_clean_xml('data/kitti/set1/AnnotationsTest', meta['labels'], exclusive = False) #ここでようやくデータセット読み込み
print('datas shape is {}', len(dumps))

strlist.sort()
print(dumps)
import pdb; pdb.set_trace()


JSN = ('data/kitti/set1/PNGImagesTest/out')



print('+++++++++++++++++++++++++++++++++++++++++++++++++++')

# jsonの読み込み
cur_dir = os.getcwd()
os.chdir(JSN)
jsons = os.listdir('.')
jsons = glob.glob(str(jsons)+'*.json')
jsonsdatasize = len(jsons)
resultBox = [0 for re2 in range(jsonsdatasize)]
for i, file in enumerate(jsons):
    with open(file) as f:
       js = json.load(f)
       jnum = len(js)
       cdBox = [[0 for ii in range(7)] for iii in range(jnum)]
       #import pdb; pdb.set_trace()
       for j in range(jnum):
           cdBox[j][0] = js[j]["label"]
           cdBox[j][6] = js[j]["confidence"]
           cdBox[j][1] = js[j]["topleft"]["x"]
           cdBox[j][2] = js[j]["topleft"]["y"]
           cdBox[j][3] = js[j]["bottomright"]["x"]
           cdBox[j][4] = js[j]["bottomright"]["y"]
           cdBox[j][5] = js[j]["dist"]
       cdBox.insert(0,int(re.sub(r'\D', '',file))) # ファイル名からどのファイルかインデックスとして抽出
    resultBox[i] = cdBox
#print(resultBox)
os.chdir(cur_dir)






"""


#img = cv2.imread('data/kitti/set1/PNGImagesTest/000002.png')
img = cv2.imread('test.jpg')
# 解析を行う
items = tfnet.return_predict(img)
# 検出できたものを確認
import pdb; pdb.set_trace()
print(items)

class_names = ['car', 'Truck']



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

        for i in class_names:
            if label == i:
                class_num = class_names.index(i)
                break

"""
