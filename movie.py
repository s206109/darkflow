import cv2

fourcc = cv2.VideoWriter_fourcc('W','M','V','1')
video = cv2.VideoWriter('video01.wmv', fourcc, 20.0, (1242, 375))
image_path = "training/image_2/"
j = 0
for i in range(0, 21):
    img = cv2.imread(image_path + '{0:06d}_{0:02d}.png'.format(j,i))
    img = cv2.resize(img, (1242,375))
    video.write(img)

video.release()
