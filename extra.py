from darkflow.net.build import TFNet
import cv2
import numpy as np
from darkflow.utils.pascal_voc_clean_xml import pascal_voc_clean_xml



#options = {"model": "cfg/yolo.cfg" ,"load":"bin/yolo.weights", "threshold": 0.1}
options = {"model": "cfg/tiny-yolo-kitti-3d.cfg" ,"load":33000, "threshold": 0.05}
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










# 画像の読み込み

img = cv2.imread('data/kitti/set1/PNGImagesTest/000002.png')
#img = cv2.imread('test.jpg')
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

        # 検出位置の表示
        """
        cv2.rectangle(img, (tlx, tly), (brx, bry), (200,200,0), 2)
        text = label + " " + ('%.2f' % dist)
        cv2.putText(img, text, (tlx+10, tly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)
        """


# 表示
cv2.imshow("View", img)
cv2.waitKey(0)
# 保存して閉じる
cv2.imwrite('out.jpg', img)
cv2.destroyAllWindows()
