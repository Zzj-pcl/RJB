from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys
import time
import random
from tqdm import tqdm
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import math

import config as config
from net import Net
from data_load import *
from train import parse_boxes_list

# 功能:使用训练好的目标检测模型对图片进行测试
def print_sentence(sent, idx2symbol):
	for tok in sent:
		tok = idx2symbol[tok]
		if tok=='<EOS>' or tok=='<pad>':
			break
		print(tok, end='')
	print('')

def print_sentences(sents, idx2symbol):
	for sent in sents:
		print_sentence(sent, idx2symbol)

def save_images(x):
	for i in range(len(x)):
		img = x[i]
		img = np.reshape(img, (img.shape[0], img.shape[1]))*255
		img = Image.fromarray(img)
		img = img.convert('RGB')
		img.save("tmp/check_{}.jpg".format(i))

def strip_eos_pad(a):
	sentinel = None
	for i in range(len(a)):
		if a[i] == config.EOS_ID or a[i] == config.PAD_ID:
			sentinel = i
			break
	return a[:sentinel]

# edit distance of a and b
def edit_dist(a, b):
	# strip pad and eos
	a = strip_eos_pad(a)
	b = strip_eos_pad(b)
	M = np.zeros((len(a)+1, len(b)+1), np.int32)
	for i in range(len(a)+1):
		M[i, 0] = i
	for j in range(len(b)+1):
		M[0, j] = j
	for i in range(1, len(a)+1):
		for j in range(1, len(b)+1):
			if a[i-1] == b[j-1]:
				M[i, j] = M[i-1, j-1]
			else:
				M[i, j] = 1+min(M[i-1, j], M[i, j-1], M[i-1][j-1])
	return float(M[len(a), len(b)])


def print_original_error():
	edist = 0
	char_count = 0
	correct_count = 0
	count = 0
	idx2symbol, symbol2idx = read_alphabet(config.alphabet_path)
	for h, o, c in tqdm(load_json_file('tmp/data_meta_check_new.json')):
		if h == o:
			correct_count += 1
		count += 1
		h = [symbol2idx[s] for s in h]
		o = [symbol2idx[s] for s in o]
		edist += edit_dist(h, o)
		char_count += len(h)
	print('total edit dist: %d' % edist)
	print('total char count: %d' % char_count)
	print('original accuracy : %f' %(1.0 - edist/char_count))
	print('original correct rate : %f' % (float(correct_count) / count))


def print_probs(probs, preds, length):
	'''
	probs: TxC float tensor
	preds: Tx1 int vector

	'''
	arr = []
	for i in range(length):
		arr.append(probs[i][preds[i]])
	print(np.array(arr, np.float32))


def print_sample(x, p, y, pred, prob, idx2symbol):
	print_sentence(x, idx2symbol)
	sentinel = 0
	for jj in range(len(p)):
		if p[jj] == 0:
			sentinel = jj
			break
	print(p[:sentinel])
	print_sentence(y, idx2symbol)
	print_sentence(pred, idx2symbol)
	sentinel = 0
	for i in range(len(pred)):
		if pred[i] == config.EOS_ID or pred[i] == config.PAD_ID:
			sentinel = i
			break
	print_probs(prob, pred, sentinel)
	print('')

def print_label(label, idx2symbol):
	s = ''
	for tok in label:
		if tok == config.EOS_ID:
			break
		s += idx2symbol[tok]
	print(s)


def nms(dets, thresh):
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]

	areas = (y2-y1+1) * (x2-x1+1)
	keep = []

	index = scores.argsort()[::-1]
	while index.size > 0:
		i = index[0]
		keep.append(i)

		x11 = np.maximum(x1[i], x1[index[1:]])
		y11 = np.maximum(y1[i], y1[index[1:]])
		x22 = np.minimum(x2[i], x2[index[1:]])
		y22 = np.minimum(y2[i], y2[index[1:]])

		w = np.maximum(0, x22-x11+1)
		h = np.maximum(0, y22-y11+1)

		overlaps = w*h
		ious = overlaps / (areas[i]+areas[index[1:]] - overlaps + 1e-10)
		
		idx = np.where(ious<=thresh)[0]
		index = index[idx+1]
	return keep

def intersection(g, p):
	g = Polygon(g[:8].reshape((4, 2)))
	p = Polygon(p[:8].reshape((4, 2)))
	if not g.is_valid or not p.is_valid:
		return 0
	inter = Polygon(g).intersection(Polygon(p)).area
	union = g.area + p.area - inter
	if union == 0:
		return 0
	else:
		return inter / union

def standard_nms(S, thresh):
	order = np.argsort(S[:, -1])[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])
		inds = np.where(ovr <= thresh)[0]
		order = order[inds+1]
	return S[keep]

def distance(x1, y1, x2, y2):
	return math.sqrt(float((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)))

def eval(save_path, total_batch, auto_vis=False):
	g = Net(config)
	g.build_net(is_training=False)
	print("Graph loaded.")

	with g.graph.as_default():
		image, Labels, GeoMaps= g.read_and_decode(save_path, is_training=False)

		sv = tf.train.Supervisor()
		with sv.managed_session() as sess:
			sv.saver.restore(sess, tf.train.latest_checkpoint(config.logdir))
			print(tf.train.latest_checkpoint(config.logdir))
			print("Restored!")

			for n in range(random.randint(0, config.num_valid_samples//config.batch_size)):
				x, _L, _G = sess.run([image, Labels, GeoMaps])

			for n in range(total_batch):
				two_pi = 2.0 * math.acos(-1)
				x, _L, _G = sess.run([image, Labels, GeoMaps])
				L_, G_ = sess.run([g.score_prob, g.geo_map], {g.x: x})
				for i in range(len(x)):
					img = x[i]
					L = L_[i]
					G = G_[i]
					#L = _L[i]
					#G = _G[i]
					L = np.reshape(L, (L.shape[0], L.shape[1]))
					img = np.reshape(img, (img.shape[0], img.shape[1], 3)) * 255
					img = Image.fromarray(255 - np.uint8(img)).convert('RGBA')
					draw = ImageDraw.Draw(img)
					#'''
					dets = []
					for r in range(L.shape[0]):
						for c in range(L.shape[1]):
							if L[r, c] > 0.618:
							#if L[r, c] > 0.4:
								tr = float(r) / float(L.shape[0])
								tc = float(c) / float(L.shape[1])
								x1 = int((tc + G[r, c, 0]*math.cos(G[r, c, 1]*two_pi))*config.max_width)
								y1 = int((tr + G[r, c, 0]*math.sin(G[r, c, 1]*two_pi))*config.max_height)
								x2 = int((tc + G[r, c, 2]*math.cos(G[r, c, 3]*two_pi))*config.max_width)
								y2 = int((tr + G[r, c, 2]*math.sin(G[r, c, 3]*two_pi))*config.max_height)
								x3 = int((tc + G[r, c, 4]*math.cos(G[r, c, 5]*two_pi))*config.max_width)
								y3 = int((tr + G[r, c, 4]*math.sin(G[r, c, 5]*two_pi))*config.max_height)
								x4 = int((tc + G[r, c, 6]*math.cos(G[r, c, 7]*two_pi))*config.max_width)
								y4 = int((tr + G[r, c, 6]*math.sin(G[r, c, 7]*two_pi))*config.max_height)
								# using triangle to filter out invalid box
								test1 = Polygon([(x1, y1), (x2, y2), (x4, y4)])
								test2 = Polygon([(x2, y2), (x3, y3), (x4, y4)])
								test3 = Polygon([(x1, y1), (x2, y2), (x3, y3)])
								test4 = Polygon([(x1, y1), (x3, y3), (x4, y4)])
								if test1.is_valid and test2.is_valid and test3.is_valid and test4.is_valid:
									edge1 = distance(x1, y1, x2, y2)
									edge2 = distance(x3, y3, x4, y4)
									edge3 = distance(x1, y1, x4, y4)
									edge4 = distance(x2, y2, x3, y3)
									if edge1>2*edge2 or edge2>2*edge1 or edge3>2*edge4 or edge4>2*edge3:
										continue
									if test1.intersection(test2).area == 0 and test3.intersection(test4).area == 0:
										score = L[r, c]
										panalty = (abs(edge1-edge2)/(edge1+edge2) + abs(edge3-edge4)/(edge3+edge4)) / 4
										score -= panalty
										dets.append([x1, y1, x2, y2, x3, y3, x4, y4, score])
								draw.point((int(tc*config.max_width), int(tr*config.max_height)), fill=(0, 255, 0, 255))
					if len(dets) > 0:
						dets = np.array(dets)
						print("\n{}_{}".format(n, i))
						print("{} boxes before nms".format(dets.shape[0]))
						keeps = standard_nms(dets, 0.1)
						print("{} boxes after nms".format(keeps.shape[0]))
						for k in range(keeps.shape[0]):
							draw.polygon(list(keeps[k][:8]), outline=(0, 255, 0, 255))
					#'''
					img.save("tmp/{}_{}_check.png".format(n, i))
					#'''
					'''
					#Image.fromarray(np.uint8(L*255)).convert('RGBA').save("tmp/{}_{}_label.png".format(n, i))
					for j in range(8):
						if j < 2:
							Image.fromarray(np.uint8(G[:, :, j]*config.mask_height*4)).convert('RGBA').save("tmp/{}_{}_geo.png".format(i, j))
						else:
							Image.fromarray(np.uint8(G[:, :, j]*config.mask_width*4)).convert('RGBA').save("tmp/{}_{}_geo.png".format(i, j))
					#'''



if __name__=='__main__':
	#'''
	os.environ['CUDA_VISIBLE_DEVICES']='1'
	start = time.time()
	eval(config.valid_tfrecords, 200)
	argv = sys.argv
	if len(argv) < 3:
		print("Usage:\n eval.py [test|valid|train] num_of_batches")
	else:
		task = argv[1]
		num = int(argv[2])
		if task == 'test':
			eval(config.test_tfrecords, num)
		elif task == 'valid':
			eval(config.valid_tfrecords, num)
		else:
			eval(config.train_tfrecords, num)
	print("Time: %f s."%(time.time()-start))
	#'''
