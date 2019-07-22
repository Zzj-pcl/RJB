import tensorflow as tf
import numpy as np
import os
import re
import cv2
import struct
from PIL import Image

import config
# 功能:制作tfrecord数据集
def resize_image(image, position):
    def scale_position(positions, rate):
        for i in range(len(positions)):
            for j in range(len(positions[i])):
                positions[i][j] = int(positions[i][j] // rate)
        return positions

    im_padded = np.ones((config.image_height,config.image_max_width), dtype=np.uint8)
    im_padded = im_padded * 255
    width, height = image.size

    width_rate = width / config.image_max_width
    height_rate = height / config.image_height
    rate = max(width_rate, height_rate)
    new_pos = scale_position(position, rate)
    new_width = int(width // rate)
    new_height = int(height // rate)
    img = image.resize((new_width, new_height))
    im_padded[:new_height, :new_width] = img.copy()
    return im_padded, new_pos

def listdir(root):
    filelist = []
    for dirpath, dirname, filename in os.walk(root):
        for filepath in filename:
            filelist.append(os.path.join(dirpath, filepath))
    return filelist


def load_data(root, save_img_path):
    file_num = 0
    # text_line_num = 0
    filelist = listdir(root)
    for filepath in filelist:
        bin_data = open(filepath, 'rb').read()
        file_num+=1
        #print(file_num)
        [dirname, filename] = os.path.split(filepath)

        offset = 0
        fmt_header = 'l8s'
        sizeofheader, format = struct.unpack_from(fmt_header, bin_data, offset)
        illu_len = sizeofheader - 36
        fmt_header = '=l8s' + str(illu_len) + 's20s2h3i'
        sizeofheader, format, illu, codetype, codelen, bits, img_h, img_w, line_num = struct.unpack_from(fmt_header,
                                                                                                         bin_data,
                                                                                                         offset)
        offset += struct.calcsize(fmt_header)

        error_flag = 0 #若文本行中存在错误label，则跳过这行
        i = 0
        while i < line_num:
            image = np.ones((img_h, img_w))
            image = image * 255
            line_word = ""
            position = np.zeros((config.label_max_len, 4), dtype=np.int32)

            fmt_line = 'i'
            word_num, = struct.unpack_from(fmt_line, bin_data, offset)
            offset += struct.calcsize(fmt_line)

            line_left = 0
            line_right = 0
            line_top = 99999
            line_down = 0
            tmp_offset = offset
            error_flag = 0
            j = 0
            i += 1
            position_num = 0
            while j < word_num:
                fmt_1 = '2s4h'
                label1, top_left_x1, top_left_y1, H, W = struct.unpack_from(fmt_1, bin_data, offset)  # 每个字符标签、左上角顶点坐标、字符图像高、宽

                if j == 0:
                    line_left = top_left_y1
                if j == word_num - 1:
                    line_right = top_left_y1 + W
                if top_left_x1 < line_top:
                    line_top = top_left_x1
                if top_left_x1 + H > line_down:
                    line_down = top_left_x1 + H

                singal_word = str(label1.decode('gbk', 'ignore').strip(b'\x00'.decode()))
                line_word += singal_word
                image_size = H * W
                offset += struct.calcsize(fmt_1)
                j += 1
                fmt_image = '=' + str(image_size) + 'B'
                images = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((H, W))
                try:
                    image[top_left_x1:top_left_x1 + H, top_left_y1:top_left_y1 + W] = images
                except:
                    print("文件名：{0}，第{1}行，第{2}个字,{3}".format(filename, i, j, singal_word))
                    print(top_left_x1, top_left_y1, H, W)
                    error_flag = 1
                offset += image_size
            if error_flag:
                continue
            '''保存position信息'''
            offset = tmp_offset
            j = 0
            position_num = 0
            while j < word_num:
                fmt_1 = '2s4h'
                label1, top_left_x1, top_left_y1, H, W = struct.unpack_from(fmt_1, bin_data, offset)  # 每个字符标签、左上角顶点坐标、字符图像高、宽

                singal_word = str(label1.decode('gbk', 'ignore').strip(b'\x00'.decode()))
                if not singal_word=="":
                    position[position_num][0] = top_left_x1-line_top
                    position[position_num][1] = top_left_y1-line_left
                    position[position_num][2] = H
                    position[position_num][3] = W
                    position_num += 1
                image_size = H * W
                offset += struct.calcsize(fmt_1)
                j += 1
                offset += image_size
            if not len(line_word)==position_num:
                print(len(line_word), position_num)
            '''保存每行'''
            image_line = image[line_top:line_down + 1, line_left:line_right + 1]
            line_file = save_img_path + filename[:-4] + '-' + str(i) + '.jpg'
            cv2.imwrite(line_file, image_line)
            im = Image.open(line_file)
            yield im, position, line_word, len(line_word)

def create_tfrecord(train_save_path, dataset_path, save_img_path):
    print("Create tfrecord")
    idx2symbol, symbol2idx = read_alphabet(config.alphabet_path)
    writer = tf.python_io.TFRecordWriter(train_save_path)
    for image, position, label, line_len in load_data(dataset_path, save_img_path):
        image, position = resize_image(image, position)
        label = re.sub('\|', ' ', label)
        label = list(label.strip())

        for i in range(len(label)):
            if 'N' + label[i] in symbol2idx:
                label[i] = 'N' + label[i]

        label = [symbol2idx[s] for s in label]
        label.append(config.EOS_ID)
        label = np.array(label, np.int32)
        if label.shape[0] >config.label_max_len or label.shape[0] <=0:
            continue
        image = np.array(image)
        image = 255 - image
        position = np.array(position)
        lfv = int(config.image_max_width / 16)
        location = np.zeros((lfv), dtype=np.float32)
        classification = np.zeros((lfv), dtype=np.int32)
        detection = np.zeros((lfv, 4), dtype=np.float32)
        grid_left = -16
        grid_right = 0
        word_idx = 0
        for j in range(lfv):
            grid_left += 16
            grid_right += 16
            center = position[word_idx][1] + position[word_idx][3] / 2  # 第word_idx个字的中心坐标
            if center>=grid_left and center<grid_right:
                idx = int(center // 16)
                location[idx] = 1
                classification[idx] = label[word_idx]
                detection[idx][0] = (position[word_idx][0]+ position[word_idx][2]/2)/ config.image_height#center
                detection[idx][1] = (center - grid_left)/ 16.0 #
                detection[idx][2] = position[word_idx][3]/config.image_height #w
                detection[idx][3] = position[word_idx][2]/config.image_height #h
                word_idx += 1
                if word_idx == label.shape[0]:
                    break

        _image_width = tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]]))
        _image = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))

        _label = [tf.train.Feature(int64_list=tf.train.Int64List(value=[tok]))for tok in label]
        print(_label)
        _label_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[label.shape[0]]))
        print(_label_length)
        _location = tf.train.Feature(bytes_list=tf.train.BytesList(value=[location.tobytes()]))
        _classification = tf.train.Feature(bytes_list=tf.train.BytesList(value=[classification.tobytes()]))
        _detection = tf.train.Feature(bytes_list=tf.train.BytesList(value=[detection.tobytes()]))
       
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'image_width': _image_width,
                'image':_image,
                'label_length': _label_length,
                'location': _location,
                'classification':_classification,
                'detection':_detection
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'label':tf.train.FeatureList(feature=_label)
            })
        )
        writer.write(example.SerializeToString())
    writer.close()
    print("tfrecord file generated.")

def save_alphabet(filepath):
    alphabet = []

    pass

def read_alphabet(filename):
    file = []
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            file.append(raw)
    idx2symbol = [s.strip('\n') for s in file]
    idx2symbol.insert(0, '<pad>')
    idx2symbol.insert(1, '<GO>')
    idx2symbol.insert(2, '<EOS>')
    idx2symbol.append(' ')
    symbol2idx = {}
    for idx, symbol in enumerate(idx2symbol):
        symbol2idx[symbol] = idx
    return idx2symbol, symbol2idx

if __name__ == '__main__':
    create_tfrecord(config.train_tfrecord, config.train_dataset_path, config.train_image_path)