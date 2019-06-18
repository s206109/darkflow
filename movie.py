import cv2

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('video.mp4', fourcc, 20.0, (640, 480))
image_path = "training/image_2"
for i in range(1, 20):
    img = cv2.imread(image_path + '000000_{0:02d}.png'.format(i))
    img = cv2.resize(img, (640,480))
    video.write(img)

video.release()
