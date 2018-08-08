# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import pdb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import math
'''
0：物体の種類（Car, Van, Truck, Pedestrian, Person_sitting, Cyclist）
1：物体の画像からはみ出している割合（0は完全に見えている、1は完全にはみ出している）
2：オクルージョン状態（0：完全に見える、1：部分的に隠れている、2：大部分が隠れている、3：不明）
3：水平方向の回転角度α[-pi, pi]
4：2D bounding boxのminx
5：2D bounding boxのminy
6：2D bounding boxのmaxx
7：2D bounding boxのmaxy
8：3D object dimensionsの高さ（height）
9：3D object dimensionsの幅（width）
10：3D object dimensionsの奥行き（length）
11：3D 物体のx座標
12：3D 物体のy座標
13：3D 物体のz座標
14：カメラ座標のy軸まわりの回転角度 [-pi..pi]
'''

visualPath = 'visualization'
imgPath = 'image_2'
nCluster = 10
nData = 1000
widthRatio = 13/1242
heightRatio = 13/370
distRatio = 13/100
alphaRatio = 13/(math.pi)


#----------------------
# データの読み込み
dir = 'label_2'
files = os.listdir(dir)
flag = 0
for file in files[:nData]:
	filePath = os.path.join(dir,file)
	print(filePath)

	# csvファイルの読み込み
	df = pd.read_csv(filePath, header=None,sep=' ')

	#1：物体の画像からはみ出している割合（0は完全に見えている、1は完全にはみ出している）
	#2：オクルージョン状態（0：完全に見える、1：部分的に隠れている、2：大部分が隠れている、3：不明）

	# Carのインデックス
	inds = np.where((df[0]=='Car') & (df[1] < 0.5) & (df[2] < 2))[0]

	# append
	tmp_minx = df[4][inds].values
	tmp_miny = df[5][inds].values
	tmp_maxx = df[6][inds].values
	tmp_maxy = df[7][inds].values
	tmp_width2d = (df[6][inds] - df[4][inds]).values
	tmp_height2d = (df[7][inds] - df[5][inds]).values
	#tmp_alpha = np.cos(df[3][inds].values)
    #もし負の数のデータの場合、πを足して向うむきに強制的に変える

	#import pdb; pdb.set_trace()
	negInds = np.where(df[3][inds].values < 0)[0]
	if  len(negInds) > 0:
         for nInd in negInds:
             df.at[inds[nInd], 3] = df[3][inds][inds[nInd]] + math.pi


	tmp_alpha = df[3][inds].values

	#tmp_ry = np.cos(df[14][inds].values)
	tmp_ry = df[14][inds].values
	tmp_height3d = df[8][inds].values
	tmp_width3d = df[9][inds].values
	tmp_length3d = df[10][inds].values
	tmp_x3d = df[11][inds].values
	tmp_y3d = df[12][inds].values
	tmp_z3d = df[13][inds].values
	tmp_fnames = np.tile(file.split('.')[0],(1,len(inds)))[0]

	if flag==0:
		minx = tmp_minx
		miny = tmp_miny
		maxx = tmp_maxx
		maxy = tmp_maxy
		width2d = tmp_width2d
		height2d = tmp_height2d
		alpha = tmp_alpha
		ry = tmp_ry
		height3d = tmp_height3d
		width3d = tmp_width3d
		length3d = tmp_length3d
		x3d = tmp_x3d
		y3d = tmp_y3d
		z3d = tmp_z3d
		fnames = tmp_fnames

		flag = 1
	else:
		fnames = np.hstack([fnames,tmp_fnames])
		minx = np.hstack([minx,tmp_minx])
		miny = np.hstack([miny,tmp_miny])
		maxx = np.hstack([maxx,tmp_maxx])
		maxy = np.hstack([maxy,tmp_maxy])
		width2d = np.hstack([width2d,tmp_width2d])
		height2d = np.hstack([height2d, tmp_height2d])
		alpha = np.hstack([alpha,tmp_alpha])
		ry = np.hstack([ry,tmp_ry])
		height3d = np.hstack([height3d,tmp_height3d])
		width3d = np.hstack([width3d,tmp_width3d])
		length3d = np.hstack([length3d,tmp_length3d])
		x3d = np.hstack([x3d,tmp_x3d])
		y3d = np.hstack([y3d,tmp_y3d])
		z3d = np.hstack([z3d,tmp_z3d])
#----------------------

#----------------------
# cluster
#kmeans= KMeans(n_clusters=nCluster, random_state=10).fit(np.vstack([width2d,height2d,minx,miny,maxx,maxy]).T)
kmeans= KMeans(n_clusters=nCluster, random_state=10).fit(np.vstack([width2d,height2d,z3d,alpha]).T)
#kmeans= KMeans(n_clusters=nCluster, random_state=10).fit(np.vstack([width2d,height2d,(maxx-minx)/2]).T)

#----------------------

#----------------------
# save cluster coordinates for anchors

# image size
im = Image.open(os.path.join(imgPath,'{}.png'.format(fnames[0])))
#im_ratio = 500.0/np.max([im.width,im.height])

cluster_centers = kmeans.cluster_centers_
x3d_mean = np.array([np.mean(x3d[np.where(kmeans.labels_==c)[0]]) for c in np.arange(nCluster)])
y3d_mean = np.array([np.mean(y3d[np.where(kmeans.labels_==c)[0]]) for c in np.arange(nCluster)])
z3d_mean = np.array([np.mean(z3d[np.where(kmeans.labels_==c)[0]]) for c in np.arange(nCluster)])


x3d_est = np.zeros(len(x3d))
y3d_est = np.zeros(len(x3d))
x3d_est_c = np.zeros(len(x3d))
y3d_est_c = np.zeros(len(x3d))

#estimate x and y
for i in np.arange(len(x3d)):
	c = kmeans.labels_[i]
	cx=(maxx[i]-minx[i])/2+minx[i]
	cy=(maxy[i]-miny[i])/2+miny[i]
	x3d_est[i] = (cx-1242/2)*z3d_mean[c]/707.04930
	y3d_est[i] = (cy-375/2)*z3d_mean[c]/707.04930
	x3d_est_c[i] = x3d_mean[c]
	y3d_est_c[i] = y3d_mean[c]

with open("cluster.pkl","wb") as fp:
	#pickle.dump(cluster_centers*im_ratio,fp)
	pickle.dump(cluster_centers,fp)
	pickle.dump(np.zeros(nCluster),fp)
	pickle.dump(np.zeros(nCluster),fp)
	pickle.dump(z3d_mean,fp)

print("img width:{}, height:{}".format(im.width,im.height))
#print(cluster_centers*im_ratio)
import pdb; pdb.set_trace()
for c in np.arange(nCluster):
	print("{},{},{},{}, ".format(round(cluster_centers[c,0]*widthRatio,1),round(cluster_centers[c,1]*heightRatio,1),round(cluster_centers[c,2]*distRatio,1), round(cluster_centers[c,3]*alphaRatio,1)))
	#print("{},{},  ".format(round(cluster_centers[c,0]*widthRatio,1),round(cluster_centers[c,1]*heightRatio,1)))

#print(cluster_centers)

print(x3d_mean)
print(y3d_mean)
print(z3d_mean)

#----------------------

#----------------------
# histogram of alpha and ry
for c in np.arange(nCluster):
	print("{}:{},{}".format(c,kmeans.cluster_centers_[c][0],kmeans.cluster_centers_[c][1]))

	inds = np.where(kmeans.labels_==c)[0]
	fig, figInd=plt.subplots(ncols=5,sharex=False)

	# rotation and coordinate
	figInd[0].plot(np.cos(alpha[inds]),np.sin(alpha[inds]),'.')
	figInd[0].set_title('alpha')
	figInd[0].set_xlim([-1,1])

	figInd[1].plot(np.cos(ry[inds]),np.sin(ry[inds]),'.')
	figInd[1].set_title('ry')
	figInd[1].set_xlim([-1,1])
	'''
	figInd[0].hist(alpha[inds])
	figInd[0].set_title('alpha')
	figInd[0].set_xlim([-3.14,3.14])

	figInd[1].hist(ry[inds])
	figInd[1].set_title('ry')
	figInd[1].set_xlim([-3.14,3.14])
	'''

	# coordinate
	figInd[2].hist(x3d[inds])
	figInd[2].set_title('x')
	figInd[2].set_xlim([-50,50])

	figInd[3].hist(y3d[inds])
	figInd[3].set_title('y')
	figInd[3].set_xlim([-50,50])

	figInd[4].hist(z3d[inds])
	figInd[4].set_title('z')
	figInd[4].set_xlim([-50,50])

	plt.savefig(os.path.join(visualPath,"alpha_ry_coodinate_hist_{}.png".format(c)))

	# dimension
	plt.clf()
	fig, figInd=plt.subplots(ncols=3,sharex=True)
	figInd[0].hist(height3d[inds])
	figInd[0].set_title('height')
	figInd[0].set_xlim([0,6])

	figInd[1].hist(width3d[inds])
	figInd[1].set_title('width')
	figInd[1].set_xlim([0,6])

	figInd[2].hist(length3d[inds])
	figInd[2].set_title('length')
	figInd[1].set_xlim([0,6])

	plt.savefig(os.path.join(visualPath,"dimension_hist_{}.png".format(c)))

	#----------------------
	# image crop
	for ind in np.arange(len(inds)):
		tmp_ind = inds[ind]
		im = Image.open(os.path.join(imgPath,'{}.png'.format(fnames[tmp_ind])))
		im_crop = im.crop((minx[tmp_ind], miny[tmp_ind], maxx[tmp_ind], maxy[tmp_ind]))
		imgVisualPath = os.path.join(visualPath,str(c))
		if not os.path.exists(imgVisualPath): os.mkdir(imgVisualPath)

		im_crop.save(os.path.join(imgVisualPath,"{}_{}.png".format(fnames[tmp_ind],ind)), quality=95)
	#----------------------
#----------------------

#----------------------
with open(os.path.join(visualPath,'log.pkl'),'wb') as fp:
	pickle.dump(kmeans,fp)
	pickle.dump(fnames,fp)
	pickle.dump(minx,fp)
	pickle.dump(miny,fp)
	pickle.dump(maxx,fp)
	pickle.dump(maxy,fp)
	pickle.dump(width2d,fp)
	pickle.dump(height2d,fp)
	pickle.dump(alpha,fp)
	pickle.dump(ry,fp)
	pickle.dump(height3d,fp)
	pickle.dump(width3d,fp)
	pickle.dump(length3d,fp)
	pickle.dump(x3d,fp)
	pickle.dump(y3d,fp)
	pickle.dump(z3d,fp)
#----------------------
