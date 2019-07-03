from darkflow.net.build import TFNet
import cv2
import numpy as np
import colorsys
import random
import time
import math
# options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1, "gpu": 0.8}
options = {"model": "cfg/tiny-yolo-kitti-3d-10.cfg", "load": 33000, "threshold": 0.1, "gpu": 1.0, "labels":"labels_kitti.txt"}
tfnet = TFNet(options)

# 動画の読み込み
cap = cv2.VideoCapture('visual_video/sample001.avi')

# 動画保存の設定
fps = 20
size = (1242, 374)
fourcc = cv2.VideoWriter_fourcc(*'H264')
# out = cv2.VideoWriter('output_tiny_yolo_voc.mp4', fourcc, fps,size)
out = cv2.VideoWriter('output_yolo.avi', fourcc, fps,size)

class_names = ['car', 'negative']

num_classes = len(class_names)

# 色リストの作成
hsv_tuples = [(x / 80, 1., 1.) for x in range(80)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.

periods = []
count = 0
while True:
    # フレームを取得
    ret, frame = cap.read()
    if not ret:
        break
    # 検出
    start = time.time()
    items = tfnet.return_predict(frame)
    count += 1
    period = time.time() - start
    if count % 30 == 0:
        print('FrameRate:' + str(1.0 / (sum(periods)/count)))

    periods.append(period)
    for item in items:
        tlx = item['topleft']['x']
        tly = item['topleft']['y']
        brx = item['bottomright']['x']
        bry = item['bottomright']['y']
        label = item['label']
        conf = item['confidence']
        dist = item['distance']

        # 自信のあるものを表示
        if conf > 0.4:

            for i in class_names:
                if label == i:
                    class_num = class_names.index(i)
                    break

            # 検出位置の表示
            dis = dist/40
            if   dis >= 0 and dis <= 0.25:
                    heatmap1 = 255
                    heatmap2 = 255 * math.sin(dis *2 * math.pi)
                    #heatmap2 = dis / 0.25*255
                    heatmap3 = 0

            elif dis > 0.25 and dis <= 0.5:
                    heatmap1 = 255 * math.sin(dis * 2 * math.pi)
                    #heatmap1 = 510 - (dis / 0.25*255)
                    heatmap2 = 255
                    heatmap3 = 0

            elif dis > 0.5 and dis <= 0.75:
                    heatmap1 = 0
                    heatmap2 = 255
                    heatmap3 = 255 * math.sin(dis * 2 * math.pi)
                    #heatmap3 = dis / 0.25 * 255 - 510

            else:
                    heatmap1 = 0
                    heatmap2 = 255 * math.sin(dis * 2 * math.pi)
                    #heatmap2 = 1020 - dis / 0.25 * 255
                    heatmap3 = 255


            cv2.rectangle(frame, (tlx, tly), (brx, bry), (heatmap3,heatmap2,heatmap1), 2)
            text = label + " " + ('%.2f' % dist)
            cv2.putText(frame, text, (tlx+10, tly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (heatmap3,heatmap2,heatmap1), 2)


    # 保存
    out.write(frame)
    # qで終了

cap.release()
out.release()
cv2.destroyAllWindows()
