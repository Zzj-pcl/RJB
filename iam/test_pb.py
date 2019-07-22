import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import config
import math
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pylab as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image, ImageDraw

# 功能:使用制作好的pb模型进行目标区域检测


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

if __name__=='__main__':
    pb_file_path = "./pb/frozen_model.pb"
    image_path = 'G:/why_workspace/RJB/detection_recognition/rep_2/img_dir/'
    image_list, raw_image_list, rate_list = get_images(image_path)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
    sess = tf.Session(config=config)
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for n in range(len(image_list)):
                gx = sess.graph.get_tensor_by_name('image_batch:0')
                score_prob = sess.graph.get_tensor_by_name('prob:0')
                geo_map = sess.graph.get_tensor_by_name('geo:0')
                x = image_list[n]
                raw_x = raw_image_list[n]
                two_pi = 2.0 * math.acos(-1)
                L_, G_ = sess.run([score_prob, geo_map], {gx: x})
                for i in range(len(x)):
                    img = x[i]
                    L = L_[i]
                    G = G_[i]
                    L = np.reshape(L, (L.shape[0], L.shape[1]))
                    img = np.reshape(img, (img.shape[0], img.shape[1], 3)) * 255
                    img = Image.fromarray(255 - np.uint8(img)).convert('RGBA')
                    draw = ImageDraw.Draw(raw_x)
                    max_width = 512 * rate_list[n]
                    max_height = 512 * rate_list[n]
                    dets = []
                    for r in range(L.shape[0]):
                        for c in range(L.shape[1]):
                            if L[r, c] > 0.618:
                                tr = float(r) / float(L.shape[0])
                                tc = float(c) / float(L.shape[1])
                                x1 = int((tc + G[r, c, 0] * math.cos(G[r, c, 1] * two_pi)) * max_width)
                                y1 = int((tr + G[r, c, 0] * math.sin(G[r, c, 1] * two_pi)) * max_height)
                                x2 = int((tc + G[r, c, 2] * math.cos(G[r, c, 3] * two_pi)) * max_width)
                                y2 = int((tr + G[r, c, 2] * math.sin(G[r, c, 3] * two_pi)) * max_height)
                                x3 = int((tc + G[r, c, 4] * math.cos(G[r, c, 5] * two_pi)) * max_width)
                                y3 = int((tr + G[r, c, 4] * math.sin(G[r, c, 5] * two_pi)) * max_height)
                                x4 = int((tc + G[r, c, 6] * math.cos(G[r, c, 7] * two_pi)) * max_width)
                                y4 = int((tr + G[r, c, 6] * math.sin(G[r, c, 7] * two_pi)) * max_height)
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
                                    if edge1 > 2 * edge2 or edge2 > 2 * edge1 or edge3 > 2 * edge4 or edge4 > 2 * edge3:
                                        continue
                                    if test1.intersection(test2).area == 0 and test3.intersection(test4).area == 0:
                                        score = L[r, c]
                                        panalty = (abs(edge1 - edge2) / (edge1 + edge2) + abs(edge3 - edge4) / (
                                                    edge3 + edge4)) / 4
                                        score -= panalty
                                        dets.append([x1, y1, x2, y2, x3, y3, x4, y4, score])
                    if len(dets) > 0:
                        dets = np.array(dets)
                        print("\n{}_{}".format(n, i))
                        print("{} boxes before nms".format(dets.shape[0]))
                        keeps = standard_nms(dets, 0.146)
                        print("{} boxes after nms".format(keeps.shape[0]))
                        for k in range(keeps.shape[0]):
                            draw.polygon(list(keeps[k][:8]), outline=(0, 255, 0, 255))
                    raw_x.save("tmp/{}_{}_check.png".format(n, i))