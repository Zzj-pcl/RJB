import tensorflow as tf
import os
import time
from PIL import Image, ImageDraw
import numpy as np
import cv2
from crnn import Net
import config

# 训练网络核心代码
def sigmoid(x):
    '''sigmoid'''
    return 1 / (1 + np.exp(-x))

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

#训练主体代码块
def train():
    g = Net(config)
    g.build_net()
    idx2symbol, symbol2idx = g.idx2symbol,g.symbol2idx
    sv = tf.train.Supervisor(graph=g.graph)
    cfg = tf.ConfigProto()
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.9
    cfg.gpu_options.allow_growth = True
    with sv.managed_session(config=cfg) as sess:
        ckpt = tf.train.latest_checkpoint(config.logdir)
        print(ckpt)
        train_writer = tf.summary.FileWriter(config.logdir + '/train', sess.graph)
		#checkpoint加载
        if ckpt:
            sv.saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))
        MLoss = 0
        MEdit_dist = 0
        MChar_count = 0
        best_loss = 1e8
        best_acc = -1e8
        time_start = time.time()
        global_step = sess.run(g.global_step)
        for step in range(global_step, g.config.total_steps):
            train_summary, img, label, label_len, mloss, medit_dist, mchar_count, _ = sess.run(
                [g.merged_summary_op, g.x, g.label, g.label_len, g.loss, g.edit_dist, g.char_count, g.train_op],
                {g.train_stage: True})
            # 过程可视化显示，检验图像和label是否对应
            #print(label[0])
            # cv2.imshow('1.jpg', img[0])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            MLoss += mloss
            MEdit_dist += medit_dist
            MChar_count += mchar_count
            train_writer.add_summary(train_summary, step)
			# 每隔show_step输出loss, acc等信息。
            if step% config.show_step==0:
                print("step=%d, Loss=%f, Acc=%f, time=%f" % (step, MLoss/config.show_step,
                                1- MEdit_dist/MChar_count, time.time()-time_start))
                MLoss = 0
                MEdit_dist = 0
                MChar_count = 0
                time_start = time.time()
			# 每隔val_step进行验证集检验,用于调整网络参数
            if step % config.val_step==0:
                VLoss = 0
                VEdit_dist = 0
                VChar_count = 0
                vcount = 50
                for j in range(vcount):
                    vloss, vedit_dist, vchar_count = sess.run([g.loss, g.edit_dist, g.char_count],
                                                              {g.train_stage: False})
                    VLoss += vloss
                    VEdit_dist += vedit_dist
                    VChar_count += vchar_count
                VLoss /= vcount
                VAcc = 1.0 - VEdit_dist/VChar_count
                if VAcc > best_acc:
                    best_loss = VLoss
                    best_acc = VAcc
                    #if VAcc > 0.8:
                    tmp = '/step_'+str(step)+'_acc_'+str(best_acc)[:4]
                    sv.saver.save(sess, config.logdir+tmp + '/model_step_%d'%step)
                print("validation --- Loss=%f, Acc=%f, Best Loss=%f, Best Acc=%f" % (VLoss, VAcc, best_loss, best_acc))

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()