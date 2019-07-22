from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image
import random
import math

from modules import *
from data_load import *
#构建网络模型，改进了U-Net

class Net(object):
	def __init__(self, config):
		self.graph = tf.Graph()
		self.config = config
	#读取数据集
	def read_and_decode(self, save_paths, is_training=True):
		def parse_example(serialized_example):
			context_features = {
				'image_height': tf.FixedLenFeature([], dtype=tf.int64),
				'image_width': tf.FixedLenFeature([], dtype=tf.int64),
				'image': tf.FixedLenFeature([], dtype=tf.string),
				'label': tf.FixedLenFeature([], dtype=tf.string),
				'geo_map': tf.FixedLenFeature([], dtype=tf.string),
			}


			context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
				context_features=context_features,
				sequence_features=sequence_features)

			image_height = tf.cast(context_parsed['image_height'], tf.int32)
			image_width = tf.cast(context_parsed['image_width'], tf.int32)
			image = tf.decode_raw(context_parsed['image'], tf.uint8)
			label = tf.decode_raw(context_parsed['label'], tf.uint8)
			geo_map = tf.decode_raw(context_parsed['geo_map'], tf.float32)
			image = tf.reshape(image, [image_height, image_width, 3])
			image = tf.cast(image, dtype=tf.float32) / 255.0
			image = tf.image.pad_to_bounding_box(image, 
																					 0,
																					 0, 
																					 self.config.max_height, 
																					 self.config.max_width)
			label = tf.reshape(label, [self.config.mask_height, self.config.mask_width])
			geo_map = tf.reshape(geo_map, [self.config.mask_height, self.config.mask_width, 8])
			label = tf.cast(label, dtype=tf.float32)
			return image, label, geo_map

		dataset = tf.data.TFRecordDataset(save_paths)
		dataset = dataset.map(parse_example)
		dataset = dataset.repeat().shuffle(10*self.config.batch_size)
		dataset = dataset.padded_batch(self.config.batch_size, ([self.config.max_height, self.config.max_width, 3], [self.config.mask_height, self.config.mask_width], [self.config.mask_height, self.config.mask_width, 8]))

		iterator = dataset.make_one_shot_iterator()
		image, labels, geo_maps = iterator.get_next()
		return image, labels, geo_maps


	def encode(self, is_training):
		enc = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)				
		enc = tf.layers.conv2d(inputs=enc, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		#enc = tf.layers.batch_normalization(enc, is_training)
		self.p1 = enc
		enc = tf.layers.max_pooling2d(inputs=enc, pool_size=[2, 2], strides=[2, 2])
		enc = tf.layers.conv2d(inputs=enc, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		enc = tf.layers.conv2d(inputs=enc, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		#enc = tf.layers.batch_normalization(enc, is_training)
		self.p2 = enc
		enc = tf.layers.max_pooling2d(inputs=enc, pool_size=[2, 2], strides=[2, 2])
		enc = tf.layers.conv2d(inputs=enc, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		enc = tf.layers.conv2d(inputs=enc, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		#enc = tf.layers.batch_normalization(enc, is_training)
		self.p3 = enc
		enc = tf.layers.max_pooling2d(inputs=enc, pool_size=[2, 2], strides=[2, 2])
		enc = tf.layers.conv2d(inputs=enc, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		enc = tf.layers.conv2d(inputs=enc, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		#enc = tf.layers.batch_normalization(enc, is_training)
		self.p4 = enc
		enc = tf.layers.max_pooling2d(inputs=enc, pool_size=[2, 2], strides=[2, 2])
		enc = tf.layers.conv2d(inputs=enc, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		enc = tf.layers.conv2d(inputs=enc, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		#enc = tf.layers.batch_normalization(enc, is_training)
		self.p5 = enc
		enc = tf.layers.max_pooling2d(inputs=enc, pool_size=[2, 2], strides=[2, 2])
		enc = tf.layers.conv2d(inputs=enc, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		enc = tf.layers.conv2d(inputs=enc, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		#enc = tf.layers.batch_normalization(enc, is_training)
		self.p6 = enc
		enc = tf.layers.max_pooling2d(inputs=enc, pool_size=[2, 2], strides=[2, 2]) 
		enc = tf.layers.conv2d(inputs=enc, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		enc = tf.layers.conv2d(inputs=enc, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		self.p7 = enc
		enc = tf.layers.max_pooling2d(inputs=enc, pool_size=[2, 2], strides=[2, 2]) 
		enc = tf.layers.conv2d(inputs=enc, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		enc = tf.layers.conv2d(inputs=enc, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		self.p8 = enc
		enc = tf.layers.max_pooling2d(inputs=enc, pool_size=[2, 2], strides=[2, 2]) 
		enc = tf.layers.conv2d(inputs=enc, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		enc = tf.layers.conv2d(inputs=enc, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		self.p9 = enc
		enc = tf.layers.max_pooling2d(inputs=enc, pool_size=[2, 2], strides=[2, 2]) 
		enc = tf.layers.conv2d(inputs=enc, filters=1024, kernel_size=(1, 1), padding='same', activation=tf.nn.relu)
		enc = tf.layers.conv2d(inputs=enc, filters=1024, kernel_size=(1, 1), padding='same', activation=tf.nn.relu)
		return enc

	def auto_decode(self, enc, is_training):
		def deconv(x, filters, output_shape, stride, name):
			w_shape = [stride, stride, filters, x.shape[-1]]
			w = tf.get_variable(name=name, shape=w_shape)
			y = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, stride, stride, 1])
			return tf.nn.leaky_relu(y)
		tshape = enc.get_shape().as_list()
		th = tshape[1]
		tw = tshape[2]
		enc = deconv(enc, 512, [self.config.batch_size, th*2, tw*2, 512], 2, 'deconv_0_w')
		#enc += self.p9
		enc = tf.concat([enc, self.p9], -1)
		enc = tf.layers.conv2d(enc, 512, [1, 1], padding='same', activation=tf.nn.leaky_relu)
		enc = tf.layers.conv2d(enc, 512, [3, 3], padding='same', activation=tf.nn.leaky_relu)
		enc = deconv(enc, 512, [self.config.batch_size, th*4, tw*4, 512], 2, 'deconv_1_w')
		#enc += self.p8
		enc = tf.concat([enc, self.p8], -1)
		enc = tf.layers.conv2d(enc, 512, [1, 1], padding='same', activation=tf.nn.leaky_relu)
		enc = tf.layers.conv2d(enc, 512, [3, 3], padding='same', activation=tf.nn.leaky_relu)
		enc = deconv(enc, 512, [self.config.batch_size, th*8, tw*8, 512], 2, 'deconv_2_w')
		#enc += self.p7
		enc = tf.concat([enc, self.p7], -1)
		enc = tf.layers.conv2d(enc, 512, [1, 1], padding='same', activation=tf.nn.leaky_relu)
		enc = tf.layers.conv2d(enc, 512, [3, 3], padding='same', activation=tf.nn.leaky_relu)
		enc = deconv(enc, 256, [self.config.batch_size, th*16, tw*16, 256], 2, 'deconv_3_w')
		#enc += self.p6
		enc = tf.concat([enc, self.p6], -1)
		enc = tf.layers.conv2d(enc, 256, [1, 1], padding='same', activation=tf.nn.leaky_relu)
		enc = tf.layers.conv2d(enc, 256, [3, 3], padding='same', activation=tf.nn.leaky_relu)
		enc = deconv(enc, 256, [self.config.batch_size, th*32, tw*32, 256], 2, 'deconv_4_w')
		#enc += self.p5
		enc = tf.concat([enc, self.p5], -1)
		enc = tf.layers.conv2d(enc, 256, [1, 1], padding='same', activation=tf.nn.leaky_relu)
		enc = tf.layers.conv2d(enc, 256, [3, 3], padding='same', activation=tf.nn.leaky_relu)
		enc = deconv(enc, 256, [self.config.batch_size, th*64, tw*64, 256], 2, 'deconv_5_w')
		#enc += self.p4
		enc = tf.concat([enc, self.p4], -1)
		enc = tf.layers.conv2d(enc, 256, [1, 1], padding='same', activation=tf.nn.leaky_relu)
		enc = tf.layers.conv2d(enc, 256, [3, 3], padding='same', activation=tf.nn.leaky_relu)
		enc = deconv(enc, 128, [self.config.batch_size, th*128, tw*128, 128], 2, 'deconv_6_w')
		#enc += self.p3
		enc = tf.concat([enc, self.p3], -1)
		enc = tf.layers.conv2d(enc, 128, [1, 1], padding='same', activation=tf.nn.leaky_relu)
		enc = tf.layers.conv2d(enc, 128, [3, 3], padding='same', activation=tf.nn.leaky_relu)

		enc = tf.layers.conv2d(enc, 13, [1, 1], padding='same')
		geo_map = enc[:, :, :, :12]
		score = enc[:, :, :, 12:]
		return score, geo_map

	def local_loss(self, geo_map):
		'''
		local constraint loss,
		local points should have similar geometry parameters
		'''
		in_channels = geo_map.get_shape().as_list()[-1]
		kernel = tf.constant([-1, -1, -1, -1, 8, -1, -1, -1, -1], shape=[3, 3], dtype=tf.float32)
		kernel /= 8
		kernel = tf.reshape(kernel, [3, 3, 1, 1])
		kernel = tf.tile(kernel, [1, 1, in_channels, 1])
		loss = tf.nn.conv2d(geo_map, kernel, [1, 1, 1, 1], padding='SAME')
		loss /= float(in_channels)
		return loss[:, :, :, 0]

	def build_net(self, is_training=True):
		with self.graph.as_default():
			if is_training:
				self.train_stage = tf.placeholder(tf.bool, shape=()) # True if train, else valid
				train_x, train_L, train_G = self.read_and_decode(self.config.train_tfrecords)
				valid_x, valid_L, valid_G = self.read_and_decode(self.config.valid_tfrecords)
				self.x = tf.cond(self.train_stage, lambda: train_x, lambda: valid_x)
				self.L = tf.cond(self.train_stage, lambda: train_L, lambda: valid_L)
				self.G = tf.cond(self.train_stage, lambda: train_G, lambda: valid_G)
			else:
				self.x = tf.placeholder(tf.float32, shape=(self.config.batch_size, self.config.max_height, self.config.max_width, 3), name='image_batch')

			self.enc = self.encode(is_training)
			self.score, geo_map = self.auto_decode(self.enc, is_training)
			g_ro = geo_map[:, :, :, ::3]
			g_phi = geo_map[:, :, :, 1::3]
			g_offset = g_ro * tf.cos(g_phi)
			g_theta = geo_map[:, :, :, 2::3]
			self.geo_map = tf.reshape(tf.stack([g_offset, g_theta], -1), [-1, self.config.mask_height, self.config.mask_width, 8], name='geo')
			self.score_map = tf.where(self.score > 0, tf.ones_like(self.score), tf.zeros_like(self.score))
			self.score_prob = tf.nn.sigmoid(self.score, name='prob')

			if is_training:
				is_target = tf.cast(self.L, tf.float32)
				# mask loss
				score_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score, labels=tf.expand_dims(tf.cast(self.L, tf.float32), -1))
				self.score_mean_loss = tf.reduce_mean(score_loss)
				# geo loss
				G_offset = self.G[:, :, :, ::2]
				G_theta = self.G[:, :, :, 1::2]
				offset_loss = tf.sqrt(tf.square(G_offset - g_offset) + 1e-10)
				theta_loss = tf.sqrt(tf.square(G_theta - g_theta) + 1e-10)
				geo_target = tf.tile(tf.expand_dims(is_target, -1), [1, 1, 1, 4])
				self.offset_mean_loss = self.config.gamma_offset * tf.reduce_sum(offset_loss * geo_target) / tf.reduce_sum(geo_target)
				self.theta_mean_loss = self.config.gamma_theta * tf.reduce_sum(theta_loss * geo_target) / tf.reduce_sum(geo_target)

				# local loss
				_local_loss = self.config.gamma_offset * tf.sqrt(tf.square(self.local_loss(g_offset)) + 1e-10) + \
											self.config.gamma_theta * tf.sqrt(tf.square(self.local_loss(g_theta)) + 1e-10)
				self.local_mean_loss = tf.reduce_sum(_local_loss * is_target) / tf.reduce_sum(is_target)

				self.mean_loss = self.score_mean_loss + self.offset_mean_loss + self.theta_mean_loss + self.local_mean_loss

				if self.config.l2_reg > 0:
					# regularization
					self.reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.config.l2_reg),
																														tf.trainable_variables())
					self.mean_loss += self.reg
				# dynamic learning rate
				global_step = tf.Variable(0, trainable=False)
				#lr = tf.train.exponential_decay(self.config.learning_rate, global_step, 1000, 0.96, False)
				lr = self.config.learning_rate
				optimizer = tf.train.AdamOptimizer(learning_rate=lr)
				# use for batch normalization
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				with tf.control_dependencies(update_ops):
					self.train_op = optimizer.minimize(self.mean_loss, global_step=global_step)

