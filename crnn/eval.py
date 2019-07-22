import tensorflow as tf
import os, io, cv2
import time
from tensorflow.contrib.rnn import *
from tensorflow.python.framework import graph_util
from PIL import Image, ImageDraw
import numpy as np
import crnn
import random
import tqdm
import config

# 功能:使用训练好的CRNN模型对图片进行测试
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#坐标计算
def dec2ori(pre, i, lfv, H, W):
    ori = [0]*4
    x = pre[0]
    y = pre[1]
    w = pre[2]
    h = pre[3]
    wb = w * H
    hb = h * H
    xb = x * H - hb/2
    # i+sigmoid(x)*W / lfv #x - hb/2
    yb = y*16.0 + i*16 - wb/2#y - wb/2 + i*16
    ori[0] = xb
    ori[1] = yb
    ori[2] = wb
    ori[3] = hb
    return ori

def box2original(pre, i, lfv, H, W):
    ori = [0]*4
    x = pre[0]
    y = pre[1]
    w = pre[2]
    h = pre[3]
    xb = (i + sigmoid(x)) * W / lfv
    yb = sigmoid(y) * H
    wb = w * H
    hb = sigmoid(h) * H
    ori[0] = xb
    ori[1] = yb
    ori[2] = wb
    ori[3] = hb
    return ori

#NMS算法
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def eval():
    g = Net(config)
    g.build_net(is_training=False)
    print("Graph loaded.")
    idx2symbol, symbol2idx = g.idx2symbol,g.symbol2idx

    with g.graph.as_default():
        image, label, location, classification, detection, label_length, image_width = g.load_tfrecord(config.valid_tfrecord)

        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(config.logdir))
            print(tf.train.latest_checkpoint(config.logdir))
            print("Restored!")

            for k in range(config.eval_times):
                img, _label, loc, cla, det, label_len, img_w =sess.run([image, label,
                                    location, classification, detection, label_length, image_width])
                pre = sess.run([g.preds], {g.x: img})
                print('label:', _label[0])
                print('label_len:', label_len[0])
                label_t = [idx2symbol[s] for s in _label[0]]
                print('label:', label_t)
                pre_np = np.array(pre)
                pre0 = pre[0][0]
                pre0_t = [idx2symbol[s] for s in pre0]
                print('predi:', pre0_t)
                img0 = img[0] * 255
                img0 = np.array(img0, dtype=np.uint8)
                w, h, c = img0.shape
                img0 = np.reshape(img0, (w, h))
                new_img = Image.fromarray(img0, 'L')
                new_img.show()

def print_sentence(sent, idx2symbol):
    for tok in sent:
        tok = idx2symbol[tok]
        if tok=='<EOS>' or tok=='<pad>':
            break
        print(tok, end='')
    print('')

#测试单张图片
def test(img_path):
    g = crnn.Net(config)
    g.build_net(is_training=False)
    print("Graph loaded.")
    idx2symbol, symbol2idx = g.idx2symbol, g.symbol2idx
    with g.graph.as_default():
        image = Image.open(img_path)
        image = image.convert('L')
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BAYER_BG2GRAY)
        size = image.shape
        image = tf.reshape(image, [1, size[0], size[1], 1])
        rate = int(size[0])/46.0
        new_height = int(int(size[1])/rate)
        image = tf.cast(image, dtype=tf.float32) / 255.0
        image = tf.image.pad_to_bounding_box(image, 0, 0, config.image_height, config.image_max_width)
        #过程可视化，验证图片预处理是否正确
		# sv = tf.train.Supervisor()
        # with sv.managed_session() as sess:
        #     image = sess.run(image)
        #     print(len(image))
        #     cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        #     cv2.imshow("Image", image[0])
        #     cv2.resizeWindow("Image", config.image_max_width, config.image_height)
        #     cv2.waitKey(0)
        _labels = tf.placeholder(tf.int32, [None])
        _label_lengths = tf.placeholder(tf.int32, [None])
        batch_label_length = tf.reduce_max(_label_lengths)
        _reshaped_labels = tf.reshape(_labels, [1, batch_label_length])
        indices = tf.where(tf.less(tf.cast(0, tf.int32), _reshaped_labels))
        values = _labels
        mask = tf.cast(tf.sign(values), dtype=tf.bool)
        values = tf.boolean_mask(values, mask)
        pmask = tf.sign(tf.abs(tf.reduce_sum(g.logits, -1)))
        sequence_length = tf.reduce_sum(tf.cast(pmask, tf.int32), -1)
        time_major_logits = tf.transpose(g.logits, [1, 0, 2])
        preds = tf.nn.ctc_greedy_decoder(time_major_logits, sequence_length)
        preds_sparse = tf.cast(preds[0][0], tf.int32)
        preds_dense = tf.sparse_to_dense(preds_sparse.indices, preds_sparse.dense_shape, preds_sparse.values)
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(config.logdir))
            print(tf.train.latest_checkpoint(config.logdir))
            print("Restored!")
            x = sess.run(image)
            p = sess.run(g.preds, {g.x: x})
            out_pb_path = "./pb2/frozen_model.pb"
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["pred"])
            with tf.gfile.FastGFile(out_pb_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            for i in range(len(p)):
                print_sentence(p[i], idx2symbol)

#测试数据集
def test_tfrecord():
    g = crnn.Net(config)
    g.build_net(is_training=False)
    print("Graph loaded.")
    idx2symbol, symbol2idx = g.idx2symbol, g.symbol2idx
    with g.graph.as_default():
        image, label, label_length = g.load_tfrecord(config.valid_tfrecord)
        sv = tf.train.Supervisor(graph=g.graph)
        cfg = tf.ConfigProto()
        cfg.gpu_options.per_process_gpu_memory_fraction = 0.8
        cfg.gpu_options.allow_growth = True
        with sv.managed_session(config=cfg) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(config.logdir))
            print(tf.train.latest_checkpoint(config.logdir))
            print("Restored!")
            for k in range(config.eval_times):
                img, _label, label_len = sess.run([image, label, label_length])
                pre = sess.run([g.preds], {g.x: img})
                print('label:', _label[0])
                print('pre_label:', pre[0])
                print('label_len:', label_len[0])
                label_t = [idx2symbol[s] for s in _label[0]]
                print('label:', label_t)
                pre_np = np.array(pre)
                pre0 = pre[0][0]
                pre0_t = [idx2symbol[s] for s in pre0]
                print('predi:', pre0_t)
                img0 = img[0] * 255
                img0 = np.array(img0, dtype=np.uint8)
                w, h, c = img0.shape
                img0 = np.reshape(img0, (w, h))
                new_img = Image.fromarray(img0, 'L')
                new_img.show()

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    path = 'G:/why_workspace/RJB/dataset2/images_enhance/119909.png' #单张图片路径
    test(path)
    #test_tfrecord()