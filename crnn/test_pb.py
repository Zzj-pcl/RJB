import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import config
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image

pb_file_path = "./pb2/frozen_model.pb"
# 功能:使用制作好的pb模型 进行卡号识别

image = Image.open('G:\why_workspace\RJB\detection_recognition/rep_2\crop/0/0.png').convert('L') #图片路径

image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BAYER_BG2GRAY)
size = image.shape
image = tf.reshape(image, [1, size[0], size[1], 1])
rate = size[0]/46
image = tf.image.resize_images(image, (46, int(size[1]/rate)))


image = tf.cast(image, dtype=tf.float32) / 255.0

image = tf.image.pad_to_bounding_box(image, 0, 0, 128, 640)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
sess = tf.Session(config=config)
with gfile.FastGFile(pb_file_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input_x = sess.graph.get_tensor_by_name('image_batch:0')       
        pred1 = sess.graph.get_tensor_by_name('pred:0')
        print(pred1.shape)
        index = sess.run(pred1, {input_x: sess.run(image)})
        print(index)