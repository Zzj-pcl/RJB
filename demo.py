import os, io
import math
import cv2
import sys
import shutil
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pylab as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image, ImageDraw
from tensorflow.python.platform import gfile

def resize_image(image):
	width, height = image.size
	rate_height = float(height) / float(512)
	rate_width = float(width) / float(512)
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
    image_list, raw_image_list, rate_list, path_list = [], [], [], []
    for image_path in files_from_folder(image_dir):
        if not os.path.exists(image_path):
            continue
        #print(image_path)
        path_list.append(image_path)
        image = Image.open(image_path)
        raw_image_list.append(image)
        image, rate = resize_image(image)
        rate_list.append(rate)
        image = np.array(image).astype(np.float32)
        if image.shape[-1] == 4:
            image = image[:, :, :-1]
        image = (255 - image) / 255.0
        image = np.expand_dims(image, 0)
        #print('1:',image.shape[1])
        #print('0', image.shape[2])
        image = np.pad(image, ((0,0), (0, 512-image.shape[1]), (0, 512-image.shape[2]), (0, 0)), 'constant')
        image_list.append(image)
    return image_list, raw_image_list, rate_list, path_list

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

def crop_rec(path, x1, y1, x2, y2, x3, y3, x4, y4):
    image = cv2.imread(path)
    b = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
    roi_t = []
    for i in range(4):
        roi_t.append(b[i])
    roi_t = np.asarray(roi_t)
    roi_t = np.expand_dims(roi_t, axis=0)
    im = np.zeros(image.shape[:2], dtype="uint8")
    cv2.polylines(im, roi_t, 1, 255)
    cv2.fillPoly(im, roi_t, 255)
    mask = im
    # cv2.imshow("Mask", mask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow("Mask to Image", masked)
    # imp = Image.fromarray(image)

    # max_w = x2 - x1
    # if x3 - x4 > max_w:
    #     max_w = x3 - x4
    # max_h = int(math.fabs(y4 - y1))
    # if math.fabs(y3 - y2) > max_h:
    #     max_h = int(math.fabs(y3 - y2))
    new_x1 = x1
    if x4 < new_x1:
        new_x1 = x4
    new_x2 = x2
    if x3 > new_x2:
        new_x2 = x3
    new_y1 = y1
    if y2 < new_y1:
        new_y1 = y2
    new_y2 = y3
    if y4 > new_y2:
        new_y2 = y4
    # print('new_x1', new_x1)
    # print('new_x2', new_x2)
    # print('new_y1', new_y1)
    # print('new_y2', new_y2)
    # print(masked.shape[0])
    # print(masked.shape[1])
    new_masked = masked[new_y1:new_y2, new_x1:new_x2]
    max_h = new_y2 - new_y1
    max_w = new_x2 - new_x1
    # print(max_h)
    # print(max_w)
    # array = np.zeros((masked.shape[0], masked.shape[1], 4), np.uint8)
    array = np.zeros((max_h, max_w, 4), np.uint8)
    array[:, :, 0:3] = new_masked
    array[:, :, 3] = 0
    array[:, :, 3][np.where(array[:, :, 0] > 2)] = 255
    array[:, :, 3][np.where(array[:, :, 1] > 2)] = 255
    array[:, :, 3][np.where(array[:, :, 2] > 2)] = 255
    return array

path_list = []
angle_list = []
if __name__=='__main__':
    g1 = tf.Graph()
    g2 = tf.Graph()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst', default='./crop/', type=str)
    parser.add_argument('--image_path', default='./img_dir/', type=str)
    parser.add_argument('--gpu', default='-1', type=int)
    args = parser.parse_args()
    # for root, dirs, files in os.walk(args.dst):
    #     for dir in dirs:
            #print(root+dir)
    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
            #os.rmdir(root+dir)
    pb_file_path = "./pb/frozen_model.pb"
    count_pic = 0

    #for root, dirs, files in os.walk(args.image_path):
        #for file in files:
            #print(root+'/'+file)
            #src = root+'/'+file
            #dst = root+'/'+'%04d' % count_pic+'.png'
            #if os.path.exists(dst):
            #    break
            #os.rename(src, dst)
            #count_pic += 1

    image_list, raw_image_list, rate_list, path_list = get_images(args.image_path)
    #if not os.path.exists(args.dst):
    os.mkdir(args.dst)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
    sess = tf.Session(config=config, graph=g1)
    with sess:
        with gfile.FastGFile(pb_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')  # 导入计算图
            gx = sess.graph.get_tensor_by_name('image_batch:0')
            score_prob = sess.graph.get_tensor_by_name('prob:0')
            for n in range(len(image_list)):
                geo_map = sess.graph.get_tensor_by_name('geo:0')
                x = image_list[n]
                img_single_path=path_list[n]
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
                        keeps = standard_nms(dets, 0.146)
                        if not os.path.exists(args.dst+'/'+str(n)):
                            os.mkdir(args.dst+'/'+str(n))
                        for k in range(keeps.shape[0]):
                            x1 = int(keeps[k][:8][0])
                            y1 = int(keeps[k][:8][1])
                            x2 = int(keeps[k][:8][2])
                            y2 = int(keeps[k][:8][3])
                            x3 = int(keeps[k][:8][4])
                            y3 = int(keeps[k][:8][5])
                            x4 = int(keeps[k][:8][6])
                            y4 = int(keeps[k][:8][7])
                            arr = crop_rec(img_single_path, x1, y1, x2, y2, x3, y3, x4, y4)
                            angle_list.append(str(x1))
                            image_1 = Image.fromarray(arr)
                            image_1.save(args.dst+'/'+str(n)+'/0'+str(k)+".png")
                print('The {}th image\'s detection finished'.format(n))
                angle_list = list(map(int, angle_list))
                templist = angle_list
                for m in range(len(templist)):
                    minlabel = int(templist.index(min(angle_list)))
                    angle_list[minlabel]=100000
                    if os.path.exists(args.dst+'/'+str(n)+'/'+str(m)+".png"):
                        os.remove(args.dst+'/'+str(n)+'/'+str(m)+".png")
                    os.rename(args.dst+'/'+str(n)+'/0'+str(minlabel)+".png", args.dst+'/'+str(n)+'/'+str(m)+".png")
                angle_list = []
            print('All images\' detection finished.')

    pb_file_path = "./pb_crnn/frozen_model.pb"
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
    sess2 = tf.Session(config=config, graph=g2)
    with sess2:
        #crop_dir =
        with gfile.FastGFile(pb_file_path, 'rb') as f:
            graph_def2 = tf.GraphDef()
            graph_def2.ParseFromString(f.read())
            tf.import_graph_def(graph_def2, name='')  # 导入计算图
            input_x = sess2.graph.get_tensor_by_name('image_batch:0')
            pred1 = sess2.graph.get_tensor_by_name('pred:0')
            write_str=''
            for i in range(len(path_list)):
                list_number = ''
                for root, dirs, files in os.walk(args.dst+'/'+str(i)):
                    for file in files:
                        image = Image.open(root+'/'+file).convert('L')
                        rate = image.size[1]/46
                        new_width = int(image.size[0]/rate)
                        image = image.resize((new_width, 46), Image.ANTIALIAS)
                        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BAYER_BG2GRAY)
                        size = image.shape
                        image = tf.reshape(image, [1, size[0], size[1], 1])
                        image = tf.cast(image, dtype=tf.float32) / 255.0
                        image = tf.image.pad_to_bounding_box(image, 0, 0, 128, 960)
                        #image = tf.image.pad_to_bounding_box(image, 0, 0, 128, 640)
                        index = sess2.run(pred1, {input_x: sess2.run(image)})[0]
                        txtname = args.dst+'/result.txt'
                        for c in range(len(index)):
                            list_number += str(int(index[c]-3))
                        list_number += '_'
                if os.path.exists(txtname) & (i == 0):
                    os.remove(txtname)
                #print(str(path_list[i]) + ' ' + str(list_number))
                #write_str+=str(path_list[i]) + ' ' + str(list_number) + '\n'
                with open(txtname, 'a', encoding='utf-8') as f2:
                    f2.write(str(path_list[i][10:]) + ':' + str(list_number[:-1]) + '\n')
                    f2.close()
            print('Iam finsihed.')
