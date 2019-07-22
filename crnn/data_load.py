from __future__ import print_function
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os, io
from PIL import Image
import collections
import random
import time
import re

import config
'''
	功能:将jpg或png格式的数据集转为tfrecord
'''
def info_file_lines(filename):
	with open(filename, 'r') as f:
		while True:
			line = f.readline().strip()
			if line:
				if line[0]=='#':
					continue
				yield line
			else:
				return

def image_path_from(token):
	tags = token.split('-')
	if len(tags) == 3:
		root_dir = config.root_dir
		L1_dir = tags[0]
		L2_dir = '-'.join(tags[0:2])
		filename = os.path.join(root_dir, L1_dir, L2_dir, token+".png")
		if os.path.exists(filename):
			return filename
	return None

# 读取图片和label信息
def samples():
	with io.open('G:/why_workspace/RJB/dataset2/label4.txt', 'r', encoding='utf-8') as f:
		data = f.readlines()
	for line in data:
		label = ''.join(line[11:])
		image = Image.open(config.root_dir+line[:10])
		image = image.convert('L')
		if image and label:
			yield image, label

#统计样本信息
def summary(info_file):
	count = 0
	exceed_count = 0
	max_width = 0
	max_height = 0
	max_label_length = 0
	width_dict = collections.defaultdict(int)
	for image, label in samples(info_file):
		count += 1
		width, height = image.size
		max_width = max(max_width, width)
		max_height = max(max_height, height)
		if height != config.final_height:
			new_width = float(width) * float(config.final_height) / float(height)
			new_width = int(new_width)
			width_dict[new_width] += 1
			if new_width > config.max_width:
				exceed_count += 1
				print('%d Exceed max width.'%exceed_count)
				continue
		if len(label) > max_label_length:
			max_label_length = len(label)
	print("total images: %d, max height:%d, max width: %d, max label length: %d"%(count, max_height, max_width, max_label_length))
	sorted_width = [(k, width_dict[k]) for k in sorted(width_dict.keys())]
	tmp = 0.0
	for k, v in sorted_width:
		tmp += float(v)

#保存tfrecord
def store_tfrecords():
	count = train_count = valid_count = 0
	_, symbol2idx = read_alphabet(config.alphabet_path)
	train_writer = tf.python_io.TFRecordWriter(config.train_save_path)
	valid_writer = tf.python_io.TFRecordWriter(config.valid_save_path)
	temp = 1
	for image, label in tqdm(samples(), total=config.num_all_samples):

		image = np.array(image)
		image = 255 - image
		label = list(label.strip())
		label = [symbol2idx[s] for s in label]
		label = np.array(label, np.int32)
		if label.shape[0] > config.max_len or label.shape[0] <= 0:
			print('continue:',count)
			continue
		temp += 1
		_image_height = tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]]))
		_image_width = tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]]))
		_label_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[label.shape[0]]))
		_image = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))	
		_label = [tf.train.Feature(int64_list=tf.train.Int64List(value=[tok])) for tok in label]
		example = tf.train.SequenceExample(
			context=tf.train.Features(feature={
				'image_height': _image_height,
				'image_width': _image_width,
				'label_length': _label_length,
				'image': _image
			}),
			feature_lists=tf.train.FeatureLists(feature_list={
				'label': tf.train.FeatureList(feature=_label)
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

#保存字典表
def save_alphabet(info_file, save_path):
	alphabet = []
	for line in info_file_lines(info_file):
		tokens = line.split()
		label = ' '.join(tokens[8:len(tokens)])
		for symbol in list(label):
			if not symbol in (alphabet):
				alphabet.append(symbol)
	with open(save_path, 'w') as f:
		for symbol in alphabet:
			f.write("%s "%(symbol))

#读取字典表，并padding前3位
#例:原始字典对应为 '0':0 '1':1 ... '100':100
#   padding后为 '<pad>':0 'GO':1 '<EOS>':2 '0':3 ... '100':103
def read_alphabet(filename):
	with open(filename, 'r') as f:
		raw = f.readline()
	idx2symbol = raw.split()
	idx2symbol.insert(0, '<pad>')
	idx2symbol.insert(1, '<GO>')
	idx2symbol.insert(2, '<EOS>')
	idx2symbol.append(' ')
	symbol2idx = {}
	for idx, symbol in enumerate(idx2symbol):
		symbol2idx[symbol] = idx
	return idx2symbol, symbol2idx

def test():
	a = collections.defaultdict(int)	
	for image, label in tqdm(samples(config.train_info_file)):
		w, h = image.size
		a[w] += 1
	with open('tmp/log_w.txt', 'w') as f:
		for k, v in sorted(a.items(), key=lambda item: item[0]):
			f.write(str(k)+" " + str(v) + "\n")


if __name__ == '__main__':
	start = time.time()
	store_tfrecords()
	print("Time: %f s."%(time.time()-start))
