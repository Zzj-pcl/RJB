import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
import numpy as np
from tensorflow.contrib.rnn import *

import config

'''
	网络模型
'''

#读取字典表
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
    #print()
	return idx2symbol, symbol2idx

class Net(object):
    def __init__(self, config):
        self.config = config
        self.graph = tf.Graph()
        self.idx2symbol, self.symbol2idx = read_alphabet(config.alphabet_path)
        print(self.idx2symbol)
	#读取tfrecord
	#输入:数据集路径
	#输出:图片矩阵点集信息，label具体值，label长度
    def load_tfrecord(self, tfrecord_path):
        lfv = int(config.image_max_width / 16)
        def parse_example(serialized_example):
            context_features = {
                'image_width': tf.FixedLenFeature([], dtype=tf.int64),
                'label_length': tf.FixedLenFeature([], dtype=tf.int64),
                'image': tf.FixedLenFeature([], dtype=tf.string)
            }
            sequence_features = {
                'label': tf.FixedLenSequenceFeature([], dtype=tf.int64)
            }

            context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized_example,
                context_features=context_features,
                sequence_features=sequence_features
            )
            image_width = tf.cast(context_parsed['image_width'], tf.int32)
            image = tf.decode_raw(context_parsed['image'], tf.uint8)
            label_length = tf.cast(context_parsed['label_length'], tf.int32)
            label = tf.cast(sequence_parsed['label'], tf.int32)
            image = tf.reshape(image, [46, image_width, 1])
            image = tf.image.resize_images(image, [46, 128*3])
            image = tf.cast(image, dtype=tf.float32) / 255.0
            image = tf.image.pad_to_bounding_box(image, 0, 0, config.image_height, config.image_max_width)
            return image, label, label_length

        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(parse_example)
        dataset = dataset.repeat().shuffle(1000 * config.batch_size)
        dataset = dataset.padded_batch(config.batch_size, ([config.image_height, config.image_max_width, 1], [config.label_max_len], []))
        iterator = dataset.make_one_shot_iterator()
        image, label, label_length = iterator.get_next()
        return image, label, label_length


	#网络模型定义
    def classification_branch(sel, inputs):
        conv_1 = tf.layers.conv2d(inputs=inputs, filters=256, kernel_size=(3, 3), strides=(2, 1), padding='same',
                                  activation=tf.nn.leaky_relu)
        conv_2 = tf.layers.conv2d(inputs=conv_1, filters=256, kernel_size=(3, 3), strides=(2, 1), padding='same',
                                  activation=tf.nn.leaky_relu)
        conv_3 = tf.layers.conv2d(inputs=conv_2, filters=512, kernel_size=(3, 3), strides=(2, 1), padding='same',
                                  activation=tf.nn.leaky_relu)
        feature_vectors = conv_3
        print("classification_branch feature:", feature_vectors)
        conv_4 = tf.layers.conv2d(inputs=conv_3, filters=512, kernel_size=(1, 1), strides=1, padding='same',
                                  activation=tf.nn.leaky_relu)
        print("classification_branch result:", conv_4)
        return feature_vectors, conv_4

    def conv_stage(self, inputs, out_dims, name, training):
        with tf.variable_scope(name_or_scope=name):
            conv1 = tf.layers.conv2d(inputs=inputs, filters=out_dims, kernel_size=(3, 3), padding='same',
                                     activation=tf.nn.relu)
            bn1= tf.layers.batch_normalization(conv1, training=training)
            pool1 = tf.layers.max_pooling2d(inputs=bn1, pool_size=[2, 2], strides=[2, 2])
        return pool1


    def base_net(self, training):
        "crnn"
        conv1 = self.conv_stage(self.x, 64, 'conv1',training)
        conv2 = self.conv_stage(conv1, 128, 'conv2', training)
        conv9 = tf.layers.conv2d(inputs=conv2, filters=256, kernel_size=(3, 3), strides=(2, 1), padding='same',
                                 use_bias=False, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv9, filters=256, kernel_size=(3, 3), padding='same', use_bias=False,
                                 activation=tf.nn.relu)
        bn3 = tf.layers.batch_normalization(conv3, training=training)
        conv4 = tf.layers.conv2d(inputs=bn3, filters=256, kernel_size=(3, 3), padding='same', use_bias=False,
                                 activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 1], strides=[2, 1])
        conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3), padding='same', use_bias=False,
                                 activation=tf.nn.relu)
        bn5= tf.layers.batch_normalization(conv5, training=training)
        conv6 = tf.layers.conv2d(inputs=bn5, filters=512, kernel_size=(3, 3), padding='same', use_bias=False,
                                 activation=tf.nn.relu)
        bn6 = tf.layers.batch_normalization(conv6, training=training)
        pool6 = tf.layers.max_pooling2d(inputs=bn6, pool_size=[2, 1], strides=[2, 1])
        conv7 = tf.layers.conv2d(inputs=pool6, filters=512, kernel_size=(2, 2), strides=(2, 2), use_bias=False,
                                 padding='valid', activation=tf.nn.relu)
        bn7 = tf.layers.batch_normalization(conv7, training=training)
        conv8 = tf.layers.conv2d(inputs=bn7, filters=512, kernel_size=(2, 2), strides=(2, 2), use_bias=False,
                                 padding='valid', activation=tf.nn.relu)

        return conv8

    def base_net2(self):
        "sa "
        conv1 = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])
        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=[2, 2])
        conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        conv6 = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[2, 2])
        conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3), padding='valid',
                                 activation=tf.nn.relu)
        conv8 = tf.layers.conv2d(inputs=conv7, filters=512, kernel_size=(3, 1), padding='valid',
                                 activation=tf.nn.relu)
        conv10 = tf.layers.conv2d(inputs=conv8, filters=512, kernel_size=(3, 1), padding='valid',
                                  activation=tf.nn.relu)
        conv9 = tf.layers.conv2d(inputs=conv10, filters=config.rnn_units, kernel_size=(2, 1), padding='valid',
                                 activation=tf.nn.relu)

        return conv9

    def backbone_net(self, is_training):

        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=tf.layers.batch_normalization,
                            weights_regularizer=slim.l2_regularizer(1e-5),
                            normalizer_params={'training': is_training}):
            conv_1 = slim.conv2d(self.x, 64, 3, 2)
            conv_2 = slim.conv2d(conv_1, 64, 3, 1)
            down_conv1 = slim.conv2d(self.x, 64, 3, 2)
            print(down_conv1, conv_2)
            res_1 = tf.nn.relu(down_conv1 + conv_2)

            conv_3 = slim.conv2d(res_1, 64, 3, 1)
            conv_4 = slim.conv2d(conv_3, 64, 3, 1)
            res_2 = tf.nn.relu(res_1 + conv_4)

            conv_5 = slim.conv2d(res_2, 128, 3, 2)
            conv_6 = slim.conv2d(conv_5, 128, 3, 1)
            down_conv2 = slim.conv2d(res_2, 128, 3, 2)
            res_3 = tf.nn.relu(down_conv2 + conv_6)
            print('res_3 ', res_3.shape)

            conv_7 = slim.conv2d(res_3, 128, 3, 1)
            conv_8 = slim.conv2d(conv_7, 128, 3, 1)
            res_4 = tf.nn.relu(res_3 + conv_8)

            conv_9 = slim.conv2d(res_4, 256, 3, 2)
            conv_10 = slim.conv2d(conv_9, 256, 3, 1)
            down_conv3 = slim.conv2d(res_4, 256, 3, 2)
            res_5 = tf.nn.relu(down_conv3 + conv_10)

            conv_11 = slim.conv2d(res_5, 256, 3, 1)
            conv_12 = slim.conv2d(conv_11, 256, 3, 1)
            res_6 = tf.nn.relu(res_5 + conv_12)

            conv_13 = slim.conv2d(res_6, 512, 3, 2)
            conv_14 = slim.conv2d(conv_13, 512, 3, 1)
            down_conv4 = slim.conv2d(res_6, 512, 3, 2)
            res_7 = tf.nn.relu(down_conv4 + conv_14)

            conv_15 = slim.conv2d(res_7, 512, 3, 1)
            conv_16 = slim.conv2d(conv_15, 512, 3, 1)
            res_8 = tf.nn.relu(res_7 + conv_16)
            return res_8

	#定义损失函数和交叉熵
    def Binary_Cross_Entropy(self, label, logits):
        y = label
        py = tf.nn.sigmoid(logits)
        py = tf.reduce_sum(py, 1)  # (?, lfv,1)
        py = tf.reduce_sum(py, -1)  # (?, lfv,1)
        self.loc_pre_t = py
        pos = py * y  # (?, lfv,1)
        pos_num = tf.reduce_sum(y)
        log_pos = tf.where(tf.equal(pos, 0), pos, tf.log(pos))
        log_pos = tf.reduce_sum(log_pos)
        neg = tf.where(tf.equal(label, 0), label + 1, label - label)
        neg = py * neg
        log_neg = tf.where(tf.equal(neg, 0), neg, tf.log(1 - neg))
        log_neg = tf.reduce_sum(log_neg)
        all_num = tf.reduce_sum(tf.ones_like(y))
        neg_num = all_num - pos_num
        loss = -1.0 / pos_num * log_pos + -1 / neg_num * log_neg
        lfv = config.image_max_width / 16
        loss = loss * lfv
        self.loc_bceloss = loss
        print('BCE loss', loss)
        return loss

    def location_loss(self, logits, loc_labels):

        logit = tf.reduce_sum(logits, 1)  # (?,lfv,1)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(loc_labels, -1), logits=logit)
        print('loc loss shape', loss)
        loss = tf.reduce_sum(loss, axis=-1)  # (?,lfv
        pos_num = tf.reduce_sum(loc_labels)
        all_num = tf.reduce_sum(tf.ones_like(loc_labels))
        neg_num = all_num - pos_num
        pos_loss = loss * loc_labels
        neg = tf.where(tf.equal(loc_labels, 0), loc_labels + 1, loc_labels - loc_labels)
        neg_loss = loss * neg
        loss = (tf.reduce_sum(pos_loss) / pos_num) + (tf.reduce_sum(neg_loss) / neg_num)
        return loss
	#均方差
    def mean_squared_error(self, labels, logits):
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        return loss

    def detection_loss(self, logits, labels, loc_label):

        lfv = config.image_max_width / 16
        loc = tf.cast(loc_label, tf.float32)
        loc = tf.expand_dims(loc, axis=-1)
        logit = tf.reduce_sum(logits, 1)  # (?, lfv, 4)
        logit = logit * loc
        loss = self.mean_squared_error(labels, logit) * (lfv * 4)
        print('mse loss :', loss)
        return loss

    def cross_entropy(self, logits, labels, location):
        logit = tf.nn.softmax(logits)  # (?,1,lfv, 7356)
        logit = tf.reduce_sum(logit, 1)  # (?, lfv, 7356)
        print("logits:", logit)
        label = tf.one_hot(labels, 7356)  # (? lfv, 7356)
        loc = tf.cast(location, tf.float32)
        loc = tf.expand_dims(loc, axis=-1)
        logit = logit * loc
        loss = label * tf.log(logit)  # shape=(?, 250, 7356)
        loss = loss * loc
        print('cla loss shape:', loss)
        loss = -tf.reduce_sum(loss)
        return loss

    def classification_loss(self, logits, labels, location):
        loc = tf.expand_dims(location, axis=-1)  # (?, lfv, 1)
        loc = tf.cast(loc, tf.float32)
        logit = tf.reduce_sum(logits, 1)  # (?, lfv, 7356)
        logit = logit * loc
        print("logits:", logit)
        label = tf.one_hot(labels, 7356)  # (? lfv, 7356)
        label = label * loc
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit)  # shape=(?, 250)
        print("class loss shape:", loss)
        loss = tf.reduce_sum(loss) / config.batch_size
        return loss

    def build_net(self, is_training=True):
        with self.graph.as_default():
            if is_training:
                self.train_stage = tf.placeholder(tf.bool, shape=())
                train_image, train_label, train_label_len = self.load_tfrecord(config.train_tfrecord)
                valid_image, valid_label, valid_label_len = self.load_tfrecord(config.valid_tfrecord)
                self.x = tf.cond(self.train_stage, lambda: train_image, lambda: valid_image)
                self.label = tf.cond(self.train_stage, lambda: train_label, lambda: valid_label)
                self.label_len = tf.cond(self.train_stage, lambda: train_label_len, lambda: valid_label_len)           
            else:
                self.x = tf.placeholder(tf.float32, shape=(None, config.image_height, config.image_max_width, 1), name='image_batch')
 

            enc = self.base_net(is_training)
            print('enc1:', enc)
            tshape = enc.get_shape().as_list()
            final_width = tshape[1] * tshape[2]
            enc = tf.reshape(enc, [-1, final_width, config.rnn_units])
            print('enc2:', enc)
            conv_mask = tf.sign(tf.abs(tf.reduce_sum(enc, -1)))
            conv_length = tf.reduce_sum(tf.cast(conv_mask, tf.int32), -1)
            for i in range(config.rnn_layers_num):
                _enc = tf.layers.dense(enc, config.rnn_units, use_bias=False)
                with tf.variable_scope("rnn_layer_{}".format(i)):
                    cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=config.rnn_units / 2, state_is_tuple=True)
                    cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=config.rnn_units / 2, state_is_tuple=True)
                    enc, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                 inputs=enc, dtype=tf.float32,
                                                                 time_major=False) 
                    enc = _enc + tf.concat(values=[enc[0], enc[1]], axis=-1)
            if is_training:
                enc = tf.layers.dropout(enc, 0.5)
                
            self.logits = tf.layers.dense(enc, len(self.idx2symbol) + 1)
            print('last logit shape', self.logits)
            logit_shape = self.logits.get_shape().as_list()
            
            time_major_logits = tf.transpose(self.logits, [1, 0, 2])  # max_time* batch_size * num_classes
            pmask = tf.sign(tf.abs(tf.reduce_sum(self.logits, -1)))
            
            seq_len = tf.fill([config.batch_size], logit_shape[1])
            print('seq:', seq_len)
            greedy_preds = tf.nn.ctc_greedy_decoder(time_major_logits, seq_len)
            preds_sparse = tf.cast(greedy_preds[0][0], tf.int32)
           
            self.preds = tf.sparse_to_dense(preds_sparse.indices, preds_sparse.dense_shape, preds_sparse.values, name='pred')
           
            print('preds:', self.preds)
           

            if is_training:
                # label转sparse
                batch_label_length = config.label_max_len
                spare_tensor_indices = tf.where(tf.less(tf.cast(0, tf.int32), self.label))  # 返回大于0的indices
                print('label shape', self.label)
                spare_tensor_values = tf.reshape(self.label, [config.batch_size * batch_label_length])
                mask = tf.cast(tf.less(tf.cast(0, tf.int32), spare_tensor_values), dtype=tf.bool)
                spare_tensor_values = tf.boolean_mask(spare_tensor_values, mask)
                labels_sparse = tf.SparseTensor(indices=spare_tensor_indices, values=spare_tensor_values,
                                                dense_shape=[config.batch_size, batch_label_length])
                loss = tf.nn.ctc_loss(labels=labels_sparse, inputs=self.logits, sequence_length=seq_len,
                                      time_major=False)
                self.loss = tf.reduce_mean(loss)
                
                self.global_step = tf.Variable(0, trainable=False)
                #定义学习率和优化器
                lr = config.learning_rate
                rate = tf.train.exponential_decay(lr, self.global_step, decay_steps=config.decay_steps, decay_rate=0.97,
                                                  staircase=True)
                opt = tf.train.AdamOptimizer(learning_rate=rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = opt.minimize(self.loss, global_step=self.global_step)
                # accuracy
                self.edit_dist = tf.reduce_sum(tf.edit_distance(preds_sparse, labels_sparse, False))
                self.char_count = tf.reduce_sum(self.label_len)
                tf.summary.scalar('loss', self.loss)
                self.merged_summary_op = tf.summary.merge_all()
