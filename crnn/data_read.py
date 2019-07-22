import numpy as np
import struct
import cv2
import os
from tqdm import tqdm

import config

root = config.valid_dataset_path

def listdir(root):
    filelist = []
    for dirpath, dirname, filename in os.walk(root):
        for filepath in filename:
            filelist.append(os.path.join(dirpath, filepath))
    return filelist

def read_dgr(root):
    alphabet = read_alphabet(config.alphabet_path)
    filelist = listdir(root)
    for filepath in tqdm(filelist):
        bin_data = open(filepath, 'rb').read()
        [dirname, filename] = os.path.split(filepath)

        offset = 0
        fmt_header = 'l8s'
        sizeofheader, format = struct.unpack_from(fmt_header, bin_data, offset)
        illu_len = sizeofheader - 36
        fmt_header = '=l8s' + str(illu_len)+'s20s2h3i'
        sizeofheader,format ,illu ,codetype , codelen , bits,img_h,img_w,line_num= struct.unpack_from(fmt_header, bin_data, offset)
        offset += struct.calcsize(fmt_header)

        i = 0
        while i < line_num:
            image = np.ones((img_h, img_w))
            image = image * 255
            line_word = ""

            fmt_line = 'i'
            word_num, = struct.unpack_from(fmt_line,bin_data,offset)
            offset += struct.calcsize(fmt_line)

            line_left = 0
            line_right = 0
            line_top = 99999
            line_down = 0

            j = 0
            i += 1
            while j < word_num:
                fmt_1 = '2s4h'
                label1, top_left_x1, top_left_y1, H, W = struct.unpack_from(fmt_1, bin_data, offset)  #每个字符标签、左上角顶点坐标、字符图像高、宽

                if j == 0:
                    line_left = top_left_y1
                if j == word_num-1:
                    line_right = top_left_y1+W
                if top_left_x1 < line_top:
                    line_top = top_left_x1
                if top_left_x1+H > line_down:
                    line_down = top_left_x1+H

                singal_word = str(label1.decode('gbk','ignore').strip(b'\x00'.decode()))
                line_word += singal_word

                if singal_word in alphabet:
                    pass
                else:
                    alphabet.append(singal_word)
                image_size = H * W
                offset += struct.calcsize(fmt_1)
                j += 1
                fmt_image = '=' + str(image_size) + 'B'
                images = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((H, W))
                try:
                    image[top_left_x1:top_left_x1+H, top_left_y1:top_left_y1+W] = images
                except:
                    print("文件名：{0}，第{1}行，第{2}个字,{3}".format(filename, i, j ,singal_word ))
                    print(top_left_x1, top_left_y1, H, W)
                offset += image_size
            '''保存每行'''
            image_line = image[line_top:line_down+1, line_left:line_right+1]
            line_file = config.valid_image_path +filename[:-4]+'-'+str(i)+'.png'
            cv2.imwrite(line_file, image_line)
    print("alphabet number:", len(alphabet))
    with open(config.alphabet_path, 'w') as f:
        for symbol in alphabet:
            f.write("%s "%(symbol))

def read_alphabet(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw = f.readline()
    idx2symbol = raw.split()
    return idx2symbol

if __name__ == '__main__':
    read_dgr(root)
