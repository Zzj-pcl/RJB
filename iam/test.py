from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys
import time
from tensorflow.python.framework import graph_util
import random
from tqdm import tqdm
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import math

import config as config
from net import Net
from data_load import *
from train import parse_boxes_list
# 测试代码
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

def resize_image(image):
	width, height = image.size
	rate_height = float(height) / float(config.max_height)
	rate_width = float(width) / float(config.max_width)
	rate = max(rate_height, rate_width)
	if rate <= 1:
		return image, rate
	new_height = int(float(height) / rate)
	new_width = int(float(width) / rate)
	return image.resize((new_width, new_height)), rate
	
def files_from_folder(image_dir):
	for root, dirs, files in os.walk(image_dir):
		for f in files:
			yield os.path.join(root, f)

def get_images(image_dir):
	image_list, raw_image_list, rate_list = [], [], []
	for image_path in files_from_folder(image_dir):
		if not os.path.exists(image_path):
			continue
		image = Image.open(image_path)
		raw_image_list.append(image)
		image, rate = resize_image(image)
		rate_list.append(rate)
		image = np.array(image).astype(np.float32)
		if image.shape[-1] == 4:
			image = image[:, :, :-1]
		image = (255 - image) / 255.0
		image = np.expand_dims(image, 0)
		image = np.pad(image, ((0,0), (0, config.max_height-image.shape[1]), (0, config.max_width-image.shape[2]), (0, 0)), 'constant')
		image_list.append(image)
	return image_list, raw_image_list, rate_list

def distance(x1, y1, x2, y2):
	return math.sqrt(float((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)))

def test(image_dir):
	image_list, raw_image_list, rate_list = get_images(image_dir)
	config.batch_size = 1
	g = Net(config)
	g.build_net(is_training=False)
	print("Graph loaded.")
	with g.graph.as_default():
		sv = tf.train.Supervisor()
		with sv.managed_session() as sess:
			sv.saver.restore(sess, tf.train.latest_checkpoint(config.logdir))
			print(tf.train.latest_checkpoint(config.logdir))
			print("Restored!")

			for n in range(len(image_list)):
			#for n in range(1):
				x = image_list[n]
				raw_x = raw_image_list[n]
				two_pi = 2.0 * math.acos(-1)
				L_, G_ = sess.run([g.score_prob, g.geo_map], {g.x: x})
				#to_pb
				out_pb_path = "./pb/frozen_model.pb"
				output_node_names = "geo,prob"
				#print(output_node_names.split(","))
				#print(isinstance(output_node_names.split(","), list))
				constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names.split(","))
				with tf.gfile.FastGFile(out_pb_path, mode='wb') as f:
					f.write(constant_graph.SerializeToString())
				####
				for i in range(len(x)):
					img = x[i]
					L = L_[i]
					G = G_[i]
					L = np.reshape(L, (L.shape[0], L.shape[1]))
					img = np.reshape(img, (img.shape[0], img.shape[1], 3)) * 255
					img = Image.fromarray(255 - np.uint8(img)).convert('RGBA')
					#draw = ImageDraw.Draw(img)
					draw = ImageDraw.Draw(raw_x)
					max_width = config.max_width * rate_list[n]
					max_height = config.max_height * rate_list[n]
					#'''
					dets = []
					for r in range(L.shape[0]):
						for c in range(L.shape[1]):
							if L[r, c] > 0.618:
								tr = float(r) / float(L.shape[0])
								tc = float(c) / float(L.shape[1])
								x1 = int((tc + G[r, c, 0]*math.cos(G[r, c, 1]*two_pi))*max_width)
								y1 = int((tr + G[r, c, 0]*math.sin(G[r, c, 1]*two_pi))*max_height)
								x2 = int((tc + G[r, c, 2]*math.cos(G[r, c, 3]*two_pi))*max_width)
								y2 = int((tr + G[r, c, 2]*math.sin(G[r, c, 3]*two_pi))*max_height)
								x3 = int((tc + G[r, c, 4]*math.cos(G[r, c, 5]*two_pi))*max_width)
								y3 = int((tr + G[r, c, 4]*math.sin(G[r, c, 5]*two_pi))*max_height)
								x4 = int((tc + G[r, c, 6]*math.cos(G[r, c, 7]*two_pi))*max_width)
								y4 = int((tr + G[r, c, 6]*math.sin(G[r, c, 7]*two_pi))*max_height)
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
								#draw.point((int(tc*config.max_width), int(tr*config.max_height)), fill=(0, 255, 0, 255))
								#draw.ppoint((int(tc*raw_x.size[0]), int(tr*raw_x.size[1])), fill=(0, 255, 0, 255))
					if len(dets) > 0:
						dets = np.array(dets)
						print("\n{}_{}".format(n, i))
						print("{} boxes before nms".format(dets.shape[0]))
						keeps = standard_nms(dets, 0.146)
						#keeps = standard_nms(dets, 0.4)
						print("{} boxes after nms".format(keeps.shape[0]))
						for k in range(keeps.shape[0]):
							draw.polygon(list(keeps[k][:8]), outline=(0, 255, 0, 255))
					raw_x.save("tmp/{}_{}_check.png".format(n, i))



if __name__=='__main__':
	#'''
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	start = time.time()
	argv = sys.argv
	if len(argv) < 2:
		test('./img_dir')
		print("Usage:\n test.py image_dir")
	else:
		image_dir = argv[1]
		test(image_dir)
	print("Time: %f s."%(time.time()-start))
	#'''
