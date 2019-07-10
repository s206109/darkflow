import cv2

for time in range(0,30):
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    image_path = "training/image_2/"
    video = cv2.VideoWriter('visual_video/sample{0:06d}.avi'.format(time), fourcc, 20.0, (1242, 375))
    for i in range(0, 20):
        img = cv2.imread(image_path + '{0:06d}_'.format(time) + '{0:02d}.png'.format(i))
        img = cv2.resize(img, (1242,375))
        video.write(img)

    video.release()

"""
image_path = "training/image_2/"
for j in range(0, 200):
    video = cv2.VideoWriter('video{0:03d}.mp4'.format(j), fourcc, 20.0, (1242, 375))
    for i in range(0, 21):
        img = cv2.imread(image_path + '{0:06d}_{0:02d}.png'.format(j,i))
        img = cv2.resize(img, (1242,375))
        video.write(img)

video.release()
"""
