import numpy as np
import math
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)
	#import pdb; pdb.set_trace()
	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape

	resultsForJSON = []
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		#import pdb; pdb.set_trace()
		if boxResults is None:
			continue
		if self.FLAGS.alpha:
			left, right, top, bot, mess, max_indx, confidence, dist, alph = boxResults
			#left, right, top, bot, mess, max_indx, confidence, dist, vecX, vecY = boxResults
		else:
			left, right, top, bot, mess, max_indx, confidence, dist= boxResults
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			if self.FLAGS.alpha:
				resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}, "dist": dist, "alph":alph})
				#resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}, "dist": dist, "vecX":vecX, "vecY":vecY})
			else:
				resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}, "dist": dist})
			continue

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx],thick//3)

	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)
