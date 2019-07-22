from __future__ import print_function
import math
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os, io
from PIL import Image
import collections
import random
import time
import re
import json
from shapely.geometry import Polygon, Point

import config


def json_files_from_folder(folder):
	for root, dirs, files in os.walk(folder):
		for f in files:
			yield f

def load_json_file(filename):
	with io.open(filename, 'r', encoding='utf-8') as f:
		data = json.load(f)
		return data

def image_from_file(fname):
	if not os.path.exists(fname):
		print(fname+'not exists.')
		return None
	image = Image.open(fname)
	return image

def center_point(points, reverse=False):
	'''
	return cx, xy
	'''
	assert len(points) == 8
	mean_x = mean_y = 0
	for j in range(len(points)):
		if j%2 == 0:
			mean_x += points[j]
		else:
			mean_y += points[j]
	mean_x /= 4
	mean_y /= 4
	if reverse:
		return mean_y, mean_x
	return mean_x, mean_y
			
def too_small(image, min_x, max_x, min_y, max_y):
	'''
	determin if the bounding box is too small
	'''
	width, height = image.size
	rate_height = float(height) / float(config.max_height)
	rate_width = float(width) / float(config.max_width)
	rate = max(rate_height, rate_width)
	rate = max(rate, 1.0)
	min_x = float(min_x) / rate
	max_x = float(max_x) / rate
	min_y = float(min_y) / rate
	max_y = float(max_y) / rate
	if max_x - min_x < config.char_min_width and max_y - min_y < config.char_min_height:
		return True
	else:
		return False

def line_rank(x):
	min_x, max_x, min_y, max_y = bounding_box(x['points'])
	return min(max_x-min_x, max_y-min_y)

def points_rank(points):
	tmp = sorted(zip(points[::2], points[1::2]), key=lambda x: x[0])
	# left points
	l = sorted(tmp[:2], key=lambda x: x[1])
	# right points
	r = sorted(tmp[2:], key=lambda x: x[1])
	return [l[0][0], l[0][1], r[0][0], r[0][1], r[1][0], r[1][1], l[1][0], l[1][1]]

def samples(folder):
	for json_file in json_files_from_folder(folder):
		image_file = config.img_dir + json_file[:-5] + '.jpg'
		json_file = config.gt_dir + json_file
		image = image_from_file(image_file)
		label = load_json_file(json_file)
		if not image or not label:
			continue
		positions = []
		lines = label['lines']
		for item in lines:
			if item['ignore'] == 0:
				points = item['points']
				if points[0] < 0:
					print(points)
					continue
				positions.append(points_rank(points))
		yield image, positions

def save_alphabet(folder, save_path):
	alphabet = []
	for _, chars, positions in tqdm(samples(folder), total=config.num_total_samples):
		for symbol in chars:
			if symbol not in alphabet:
				alphabet.append(symbol)
	with io.open(save_path, 'w', encoding='utf-8') as f:
		for symbol in alphabet:
			f.write("%s\n"%(symbol))

def read_alphabet(filename):
	with io.open(filename, 'r', encoding='utf-8') as f:
		raw = f.readlines()
	idx2symbol = [s.strip('\n') for s in raw]
	idx2symbol.insert(0, '<pad>')
	idx2symbol.insert(1, '<GO>')
	idx2symbol.insert(2, '<EOS>')
	idx2symbol.append(' ')
	symbol2idx = {}
	for idx, symbol in enumerate(idx2symbol):
		symbol2idx[symbol] = idx
	return idx2symbol, symbol2idx

def summary(folder):
	count = 0
	max_len = 0
	max_height = max_width = mean_height = mean_width = 0
	for image, chars, positions in tqdm(samples(folder)):
		count += 1
		width, height = image.size
		mean_height += height
		mean_width += width
		if width > max_width:
			max_width = width
		if height > max_height:
			max_height = height
		if len(chars) > max_len:
			max_len = len(chars)
	mean_height /= count
	mean_width /= count
	print('max length: %d, max height: %d, max width: %d, mean_height: %f, mean_width: %f'%(max_len, max_height, max_width, mean_height, mean_width))
	print('total samples: %d'%count)

def resize_image(image, positions):
	def scale_positions(positions, rate):
		rate = max(rate, 1.0)
		for i in range(len(positions)):
			for j in range(len(positions[i])):
				if j % 2 == 0: # x coordinate, scale with width
					positions[i][j] = float(positions[i][j]) / (rate * float(config.max_width))
				else: # y coordinate, scale width height
					positions[i][j] = float(positions[i][j]) / (rate * float(config.max_height))
		return positions

	width, height = image.size
	rate_height = float(height) / float(config.max_height)
	rate_width = float(width) / float(config.max_width)
	rate = max(rate_height, rate_width)
	positions = scale_positions(positions, rate)
	if rate <= 1:
		return image, positions
	new_height = int(float(height) / rate)
	new_width = int(float(width) / rate)
	assert new_height <= config.max_height
	assert new_width <= config.max_width
	image = image.resize((new_width, new_height))
	return image, positions


def bounding_box(points):
	'''
	return left top right bottom
	'''
	min_x, max_x, min_y, max_y = config.max_width, 0, config.max_height, 0
	for i in range(len(points)):
		if i%2 == 0:
			min_x = min(min_x, points[i])
			max_x = max(max_x, points[i])
		else:
			min_y = min(min_y, points[i])
			max_y = max(max_y, points[i])
	return min_x, min_y, max_x, max_y

def polar_coord(ox, oy, tx, ty):
	'''
	ox, oy: origion point
	tx, ty: point
	return r and theta, theta in range [0, 2*pi]
	'''
	r = math.sqrt((tx-ox)*(tx-ox) + (ty-oy)*(ty-oy))
	if r == 0:
		return 0, 0
	dx = tx - ox
	theta = math.acos(dx/r)
	if ty < oy:
		theta = 2.0*math.acos(-1) - theta
	return r, theta/math.acos(-1)/2.0
	
def store_tfrecords(data_dir, train_save_path, valid_save_path):
	count = train_count = valid_count = 0
	#_, symbol2idx = read_alphabet(config.alphabet_path)
	train_writer = tf.python_io.TFRecordWriter(train_save_path)
	valid_writer = tf.python_io.TFRecordWriter(valid_save_path)
	for image, positions in tqdm(samples(data_dir), total=config.num_total_samples):
		# resize image
		image, positions = resize_image(image, positions)
		image = np.array(image)
		image = 255 - image
		label = np.zeros((config.mask_height, config.mask_width), dtype=np.uint8)
		geo_map = np.zeros((config.mask_height, config.mask_width, 8), dtype=np.float32)
		for i in range(len(positions)):
			points = positions[i]
			left, top, right, bottom = bounding_box(points)
			left = int(left * config.mask_width)
			top = int(top * config.mask_height)
			right = int(right * config.mask_width)
			bottom = int(bottom * config.mask_height)
			# filter out too small line
			if right - left < config.line_min_width and bottom - top < config.line_min_height:
				continue
			poly = Polygon([(points[0], points[1]), (points[2], points[3]), (points[4], points[5]), (points[6], points[7])])
			for y in range(max(0, top), min(config.mask_height-1, bottom+1)):
				for x in range(max(0, left), min(config.mask_width-1, right+1)):
					tx = float(x) / float(config.mask_width)
					ty = float(y) / float(config.mask_height)
					if poly.contains(Point(tx, ty)):
						label[y, x] = 1
						for k in range(4):
							r, theta = polar_coord(tx, ty, points[2*k], points[2*k+1])
							geo_map[y, x, 2*k] = r
							geo_map[y, x, 2*k+1] = theta
		# make features
		_image_height = tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]]))
		_image_width = tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]]))
		#_label_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[chars.shape[0]]))
		_image = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
		_label = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()]))
		_geo_map = tf.train.Feature(bytes_list=tf.train.BytesList(value=[geo_map.tobytes()]))
		#_prev_map = tf.train.Feature(bytes_list=tf.train.BytesList(value=[prev_map.tobytes()]))
		#_next_map = tf.train.Feature(bytes_list=tf.train.BytesList(value=[next_map.tobytes()]))
		# label is feature list
		#_chars = [tf.train.Feature(int64_list=tf.train.Int64List(value=[tok])) for tok in chars]
		#_positions = [tf.train.Feature(float_list=tf.train.FloatList(value=pos)) for pos in positions]
		example = tf.train.SequenceExample(
			context=tf.train.Features(feature={
				'image_height': _image_height,
				'image_width': _image_width,
				#'label_length': _label_length,
				'image': _image,
				'label': _label,
				'geo_map': _geo_map,
				#'prev_map': _prev_map,
				#'next_map': _next_map
			}),
			feature_lists=tf.train.FeatureLists(feature_list={
				#'chars': tf.train.FeatureList(feature=_chars),
				#'positions': tf.train.FeatureList(feature=_positions)
			})
		)
		if count < config.num_valid_samples:
			valid_writer.write(example.SerializeToString())
			valid_count += 1
		else:
			train_writer.write(example.SerializeToString())
			train_count += 1
		count += 1
	train_writer.close()
	valid_writer.close()
	print('%d train samples generated.\n%d valid samples generated.' % (train_count, valid_count))




def test():
	#summary(config.gt_dir)
	store_tfrecords(config.gt_dir, config.train_save_path, config.valid_save_path)



if __name__ == '__main__':
	start = time.time()
	test()
	print("Time: %f s."%(time.time()-start))
