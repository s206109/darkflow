# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import pdb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import matplotlib.patches as patches

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

# 何度も色を生成したいなら関数化
def generate_random_color():
    return tuple([np.random.rand() for _ in range(3)])
    
visualPath = 'visualization3d'
imgPath = 'image_2'
nCluster = 20
nData = 1000
nSubAnchors = 1

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
#kmeans= KMeans(n_clusters=nCluster, random_state=10).fit(np.vstack([width2d,height2d]).T)
#kmeans= KMeans(n_clusters=nCluster, random_state=10).fit(np.vstack([width2d,height2d,(maxx-minx)/2]).T)

# normalization
x3d_max = np.max(np.abs(x3d))
x3d_norm = x3d/np.max(x3d_max)
width2d_max = np.max(np.abs(width2d))
width2d_norm = width2d/np.max(width2d_max)
height2d_max = np.max(np.abs(height2d))
height2d_norm = height2d/np.max(height2d_max)
z3d_max = np.max(np.abs(z3d))
z3d_norm = z3d/z3d_max
alpha_max = np.max(np.abs(alpha))
alpha_norm = alpha/alpha_max

#X3d = np.vstack([x3d_norm,z3d_norm,alpha_norm]).T
X3d = np.vstack([width2d_norm,height2d_norm,z3d_norm,alpha_norm]).T
X2d = np.vstack([minx,miny,maxx,maxy]).T
kmeans= KMeans(n_clusters=nCluster, random_state=10).fit(X3d)
#----------------------

#----------------------
# save cluster coordinates for anchors
ccs_norm = kmeans.cluster_centers_
width2d_c = ccs_norm[:,0] * width2d_max
height2d_c = ccs_norm[:,1] * height2d_max
z3d_c = ccs_norm[:,2] * z3d_max
alpha_c = ccs_norm[:,3] * alpha_max

# cluster centers
ccs = np.vstack([width2d_c,height2d_c,z3d_c,alpha_c]).T

# anchors
anchors_std = np.zeros([nCluster,4])
for c in np.arange(nCluster):
	inds = np.where(kmeans.labels_==c)[0]
	
	N = np.min([nSubAnchors,len(inds)])
	anchors_ = X2d[inds[np.random.permutation(len(inds))[:N]],:]
	if c==0:
		anchors = anchors_
	else:
		anchors = np.vstack([anchors, anchors_])
	#anchors_[c,:] = np.hstack([X2d[inds[np.argmin(np.sum(np.square(X3d[inds] - ccs_norm[c,:]),axis=1))]], z3d_c[c], x3d_c[c], np.mean(y3d)])

	anchors_std[c,:] = np.std(X2d[inds,:],axis=0)

with open("cluster.pkl","wb") as fp:
	pickle.dump(anchors,fp)
	pickle.dump(anchors_std,fp)
	pickle.dump(width2d_c,fp)
	pickle.dump(height2d_c,fp)
	pickle.dump(z3d_c,fp)
	pickle.dump(alpha_c,fp)
	pickle.dump(ccs,fp)
	
# print image size and cluster centers
im = Image.open(os.path.join(imgPath,'{}.png'.format(fnames[0])))
print("img width:{}, height:{}".format(im.width,im.height))
print(ccs)
pdb.set_trace()
#----------------------

