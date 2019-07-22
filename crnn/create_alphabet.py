import struct
import os
from tqdm import tqdm

import config

root = config.train_dataset_path

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
        offset = 0
        fmt_header = 'l8s'
        sizeofheader, format = struct.unpack_from(fmt_header, bin_data, offset)
        illu_len = sizeofheader - 36
        fmt_header = '=l8s' + str(illu_len)+'s20s2h3i'

        sizeofheader,format ,illu ,codetype , codelen , bits,img_h,img_w,line_num= struct.unpack_from(fmt_header, bin_data, offset)

        offset += struct.calcsize(fmt_header)

        i = 0
        while i < line_num:
            line_word = ""

            fmt_line = 'i'
            word_num, = struct.unpack_from(fmt_line,bin_data,offset)
            offset += struct.calcsize(fmt_line)

            j = 0
            i += 1
            while j < word_num:
                fmt_1 = '2s4h'
                label1, top_left_x1, top_left_y1, H, W = struct.unpack_from(fmt_1, bin_data, offset)  #每个字符标签、左上角顶点坐标、字符图像高、宽

                singal_word = str(label1.decode('gbk','ignore').strip(b'\x00'.decode()))
                line_word += singal_word

                if singal_word in alphabet:
                    pass
                else:
                    alphabet.append(singal_word)

                image_size = H * W
                offset += struct.calcsize(fmt_1)
                j += 1
                offset += image_size
    print("alphabet number:", len(alphabet))
    with open(config.alphabet_path, 'w') as f:
        for symbol in alphabet:
            f.write("%s "%(symbol))
#读取字典表
def read_alphabet(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            raw = f.readline()
    except:
        return []
    idx2symbol = raw.split()
    return idx2symbol

if __name__ == '__main__':
    # read_dgr(config.train_dataset_path)
    read_dgr(config.valid_dataset_path)
