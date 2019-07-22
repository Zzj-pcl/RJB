import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageDraw

import config
from net import Net
from data_load import *
#网络训练主体代码
def train():
	g = Net(config)
	g.build_net()
	sv = tf.train.Supervisor(graph=g.graph, logdir=g.config.logdir)
	cfg = tf.ConfigProto()
	cfg.gpu_options.per_process_gpu_memory_fraction = 0.9
	cfg.gpu_options.allow_growth=True
	with sv.managed_session(config=cfg) as sess:
		ckpt = tf.train.latest_checkpoint(config.logdir)
		start_step = 0
		if ckpt:
		#加载checkpoint，断点保存恢复
			sv.saver.restore(sess, ckpt)
			print("restore from the checkpoint {0}".format(ckpt))
			for root, dir, files in os.walk(config.logdir):
				for file in files:
					if file.startswith('model_step_'):
						temp = file.split('.')
						if int(temp[0][11:])>start_step:
							start_step = int(temp[0][11:])
			print('start_step=',start_step)
		best_loss = 1e8
		best_auto_loss = 1e8
		not_improve_count = 0
		MLoss = 0
		MClsLoss = 0
		MRegLoss = 0
		MAutoLoss = 0
		MAttLoss = 0
		mloss = m_cls_loss = m_reg_loss = m_auto_loss = m_att_loss = 0
		time_start = time.time()
		for st in range(start_step, g.config.total_steps):
			mloss, m_cls_loss, m_reg_loss, m_auto_loss, m_att_loss, _ = sess.run([g.mean_loss, g.score_mean_loss, g.offset_mean_loss, g.theta_mean_loss, g.local_mean_loss, g.train_op], {g.train_stage: True})
			MLoss += mloss
			MClsLoss += m_cls_loss
			MRegLoss += m_reg_loss
			MAutoLoss += m_auto_loss
			MAttLoss += m_att_loss
			# display
			if st % g.config.display == 0:
				print("step=%d, Loss=%f, score Loss=%f, offset Loss=%f, theta Loss=%f, local Loss=%f, time=%f"%(st, MLoss/g.config.display, MClsLoss/g.config.display, MRegLoss/g.config.display, MAutoLoss/g.config.display, MAttLoss/g.config.display, time.time()-time_start))
				MLoss = MClsLoss = MRegLoss = MAutoLoss = MAttLoss = 0
				time_start = time.time()
			valid_step = g.config.num_train_samples // g.config.batch_size
			#验证集验证，用于网络调参
			if st % valid_step == 0:
				VLoss = VClsLoss = VRegLoss = VAutoLoss = VAttLoss = 0
				vloss = v_cls_loss = v_reg_loss = v_auto_loss = v_att_loss = 0
				count = g.config.num_train_samples // g.config.batch_size
				for vi in range(count):
					vloss, v_cls_loss, v_reg_loss, v_auto_loss, v_att_loss = sess.run([g.mean_loss, g.score_mean_loss, g.offset_mean_loss, g.theta_mean_loss, g.local_mean_loss], {g.train_stage: False})
					VLoss += vloss
					VClsLoss += v_cls_loss
					VRegLoss += v_reg_loss
					VAutoLoss += v_auto_loss
					VAttLoss += v_att_loss
				VLoss /= count
				VClsLoss /= count
				VRegLoss /= count
				VAutoLoss /= count
				VAttLoss /= count
				print("validation --- Loss=%f, score Loss=%f, offset Loss=%f, theta Loss=%f, local Loss=%f"%(VLoss, VClsLoss, VRegLoss, VAutoLoss, VAttLoss))
				# model select && early stop
				if VLoss < best_loss or VAutoLoss < best_auto_loss:
					best_loss = VLoss
					best_auto_loss = VAutoLoss
					not_improve_count = 0
					sv.saver.save(sess, g.config.logdir+'/model_step_%d'%st)
				else:
					not_improve_count += 1
				if not_improve_count >= g.config.early_stop_count:
					print("training stopped, best Loss=%f"%(best_loss))
					break
					sv.request_stop()

def print_label(label, idx2symbol):
	s = ''
	for tok in label:
		if tok > 2:
			s = s + idx2symbol[tok]
	print(s)


def parse_boxes_list(L, G, prev_map, next_map, thresh):
	'''
	L: score_map
	G: geo_map
	'''
	boxes_list = []
	bounding_box_list = []
	dets = []
	for r in range(L.shape[0]):
		for c in range(L.shape[1]):
			if L[r, c] > 0.5:
				tr = float(r)/float(config.mask_height)
				tc = float(c)/float(config.mask_width)
				top = max(0, int((tr - G[r, c, 0]) * config.max_height))
				bottom = min(config.max_height-1, int((tr + G[r, c, 1]) * config.max_height))
				left = max(0, int((tc - G[r, c, 2]) * config.max_width))
				right = min(config.max_width-1, int((tc + G[r, c, 3]) * config.max_width))
				dets.append([left, top, right, bottom, L[r, c], r, c])
	# decoding by nms
	dets = np.array(dets)
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]
	areas = (y2-y1+1)*(x2-x1+1)
	index = scores.argsort()[::-1]
	while index.size > 0:
		# select best score box
		i = index[0]
		def search_by_map(boxes, cx, cy, G, link_map, reverse, mt, mb, ml, mr):
			'''
			mt, mb, ml, mr: bounding box top, bottom, left, right
			'''
			count = 0
			while cx >= 0 and cy >= 0:
				rcx = float(cx) / float(config.mask_width)
				rcy = float(cy) / float(config.mask_height)
				top = max(0, int((rcy - G[cy, cx, 0]) * config.max_height))
				bottom = min(config.max_height-1, int((rcy + G[cy, cx, 1]) * config.max_height))
				left = max(0, int((rcx - G[cy, cx, 2]) * config.max_width))
				right = min(config.max_width-1, int((rcx + G[cy, cx, 3]) * config.max_width))
				# select bounding box
				if top < mt:
					mt = top
				if bottom > mb:
					mb = bottom
				if left < ml:
					ml = left
				if right > mr:
					mr = right
				gcx = int(rcx * config.max_width)
				gcy = int(rcy * config.max_height)
				if left != right and top != bottom and L[cy, cx] > 0.5:
					if reverse:
						boxes.insert(0, [left, top, right, bottom, gcx, gcy])
					else:
						boxes.append([left, top, right, bottom, gcx, gcy])
				count += 1
				if count > 20:
					break
				tcx = min(int((rcx + link_map[cy, cx, 0])*config.mask_width), config.mask_width-1)
				tcy = min(int((rcy + link_map[cy, cx, 1])*config.mask_height), config.mask_height-1)
				cx = tcx
				cy = tcy
				if cx == 0 and cy == 0:
					break
			return boxes, mt, mb, ml, mr
		boxes = []
		mt, mb, ml, mr = config.max_height, 0, config.max_width, 0
		cy = int(dets[i][-2])
		cx = int(dets[i][-1])
		# prev search
		boxes, mt, mb, ml, mr = search_by_map(boxes, cx, cy, G, prev_map, True, mt, mb, ml, mr)
		# prev search
		boxes, mt, mb, ml, mr = search_by_map(boxes, cx, cy, G, next_map, False, mt, mb, ml, mr)
		# delete boxes by nms
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
		# append boxes to boxes list
		boxes_list.append(boxes)
		bounding_box_list.append([ml, mt, mr, mb])
	return boxes_list, bounding_box_list



def read_test():
	g = Net(config)
	image, labels, geo_maps = g.read_and_decode(config.train_tfrecords)
	with tf.Session() as sess:
		_image, _labels, _geo_maps = sess.run([image, labels, geo_maps])
		for i in range(len(_image)):
			img = _image[i]
			L = _labels[i]
			G = _geo_maps[i]
			img = np.reshape(img, (img.shape[0], img.shape[1], 3)) * 255
			img = Image.fromarray(255 - np.uint8(img)).convert('RGBA')
			draw = ImageDraw.Draw(img)
			img.save("tmp/{}_check.png".format(i))
			#'''
			Image.fromarray(np.uint8(L*255)).convert('RGBA').save("tmp/{}_label.png".format(i))
			for j in range(8):
				if j%2 ==  0:
					Image.fromarray(np.uint8(G[:, :, j]*255)).convert('RGBA').save("tmp/{}_{}_geo.png".format(i, j))
				else:
					Image.fromarray(np.uint8(G[:, :, j]*255)).convert('RGBA').save("tmp/{}_{}_geo.png".format(i, j))
			#'''

if __name__=='__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	#read_test()
	train()
