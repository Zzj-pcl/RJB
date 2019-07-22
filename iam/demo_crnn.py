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

# 测试代码
pb_file_path = "./pb2/frozen_model.pb"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
sess2 = tf.Session(config=config)
with gfile.FastGFile(pb_file_path, 'rb') as f:
    graph_def2 = tf.GraphDef()
    graph_def2.ParseFromString(f.read())
    #sess2.graph.as_default()
    tf.import_graph_def(graph_def2, name='')  # 导入计算图
    with sess2:
        #sess.run(tf.global_variables_initializer())
        for i in range(len(path_list)):
            list_number = ''
            for root, dirs, files in os.walk('./crop/'+str(i)):
                for file in files:
                    #print(root+'/'+file)
                    image = Image.open(root+'/'+file).convert('L')
                    #print(image.size[0])
                    #print(image.size[1])
                    rate = image.size[0]/46
                    new_width = int(image.size[1]/rate)
                    #print(new_width)
                    image = image.resize((new_width, 46), Image.ANTIALIAS)
                    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BAYER_BG2GRAY)
                    size = image.shape
                    image = tf.reshape(image, [1, size[0], size[1], 1])
                    image = tf.cast(image, dtype=tf.float32) / 255.0
                    image = tf.image.pad_to_bounding_box(image, 0, 0, 128, 640)
                    # sv = tf.train.Supervisor()
                    # with sv.managed_session() as sess:
                    #     image = sess.run(image)
                    #     #print(len(image))
                    #     #cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
                    #     cv2.imshow("Image", image[0])
                    #     #cv2.resizeWindow("Image", 128, 640)
                    #     cv2.waitKey(0)
                    #image = tf.reshape(image, [128, 128])
                    #config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
                    #tf.import_graph_def(graph_def2, name='')  # 导入计算图
                    input_x = sess2.graph.get_tensor_by_name('image_batch:0')
                    pred1 = sess2.graph.get_tensor_by_name('pred:0')
                    #print(pred1.shape)
                    index = sess2.run(pred1, {input_x: sess.run(image)})[0]
                    print(index)
                    txtname = './crop/result.txt'
                    for c in range(len(index)):
                        list_number += str(index[c]-3)
                    list_number += ' '
            print(list_number)
            with open(txtname, 'a', encoding='utf-8') as f2:
                f2.write(str(path_list[i]) + ' ' + str(list_number) + '\n')
                f2.close()
        print('Iam finsihed.')