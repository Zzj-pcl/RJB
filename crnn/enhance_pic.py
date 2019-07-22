import cv2
from math import *
import numpy as np
import os
from skimage import data, exposure, img_as_float
import json, io
from collections import defaultdict
import matplotlib as plt
# 功能:数据增强核心代码，包括旋转、亮度调整、仿射变换、压缩拉伸


#坐标计算
def cal_x(x ,y, degree, w, h):
    x1 = x / fabs(cos(radians(degree)))
    x2 = (h - y - x * fabs(tan(radians(degree)))) * fabs(sin(radians(degree)))
    return str(round(x1 + x2))

def cal_y(x, y, degree, w, h):
    y1 = (x - y * fabs(tan(radians(degree)))) * fabs(sin(radians(degree)))
    y2 = y / fabs(cos(radians(degree)))
    return str(round(y1 + y2))

def cal_x2(x, y, degree, w, h):
    x1 = (y - x * fabs(tan(radians(degree)))) * fabs(sin(radians(degree)))
    x2 = x / fabs(cos(radians(degree)))
    return str(round(x1 + x2))

def cal_y2(x ,y, degree, w, h):
    y1 = y / fabs(cos(radians(degree)))
    y2 = (w - x - y * fabs(tan(radians(degree)))) * fabs(sin(radians(degree)))
    return str(round(y1 + y2))

def count_rotate(count, new_count, h ,w, degree):
    filename = 'D:/dataset2/json/' + '%06d' % count + '.json'
    with io.open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    for d1 in data:
        d = defaultdict(list)
        x1 = int(d1[7:].split(',')[0])
        y1 = int(d1[7:].split(',')[1])
        x2 = int(d1[7:].split(',')[2])
        y2 = int(d1[7:].split(',')[5])
        if degree>0:
            d[''].append(
                cal_x2(x1, y1, degree, w, h) + ',' + cal_y2(x1, y1, degree, w, h) + ',' + cal_x2(x2, y1, degree, w, h) + ','
                + cal_y2(x2, y1, degree, w, h) + ',' + cal_x2(x2, y2, degree, w, h) + ',' + cal_y2(x2, y2, degree, w, h)
                + ',' + cal_x2(x1, y2, degree, w, h) + ',' + cal_y2(x1, y2, degree, w, h))

        else:
            d[''].append(
                cal_x(x1, y1, degree, w, h) + ',' + cal_y(x1, y1, degree, w, h) + ',' + cal_x(x2, y1, degree, w, h) + ','
                + cal_y(x2, y1, degree, w, h) + ',' + cal_x(x2, y2, degree, w, h) + ',' + cal_y(x2, y2, degree, w, h)
                + ',' + cal_x(x1, y2, degree, w, h) + ',' + cal_y(x1, y2, degree, w, h))
        json_str = json.dumps(d)
        with open('D:/dataset2/json/' + '%06d' % new_count + '.json', 'a') as f:
            f.write(json_str + '\n')

def rotate_img(img_path, degree):
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderMode=cv2.BORDER_REPLICATE)
    rate = heightNew / 46
    imgRotation = cv2.resize(imgRotation, (int(widthNew / rate), 46))
    return imgRotation, h, w

def brightness(img_path):
    img = cv2.imread(img_path)
    img1 = exposure.adjust_gamma(img, 1.25)  # 调暗
    img2 = exposure.adjust_gamma(img, 1.5)  # 调亮
    img3 = exposure.adjust_gamma(img, 0.75)  # 调暗
    img4 = exposure.adjust_gamma(img, 0.5)  # 调亮
    return img1, img2, img3, img4

def wap(img_path):
    img = cv2.imread(img_path)
    height, width= img.shape[:2]
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[0 + width*0.1, height*0.1], [width-width*0.1, height*0.1], [0, height], [width, height]])
    pts3 = np.float32([[0, 0], [width, 0], [width*0.1, height-height*0.1], [width-width*0.1, height-height*0.1]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    M2 = cv2.getPerspectiveTransform(pts1, pts3)
    res1 = cv2.warpPerspective(img, M, (width, int(height)), borderMode=cv2.BORDER_REPLICATE)
    res2 = cv2.warpPerspective(img, M2, (width, int(height)), borderMode=cv2.BORDER_REPLICATE)
    return res1, res2

def images_resize(img_path):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    res1 = cv2.resize(img, (int(width * 0.5), 46))
    res2 = cv2.resize(img, (int(width * 0.75), 46))
    res3 = cv2.resize(img, (int(width * 1.25), 46))
    res4 = cv2.resize(img, (int(width * 1.5), 46))
    return res1, res2, res3, res4


def count_json_brightness(count, new_count):
    filename = 'D:/dataset2/json/' + '%06d' % count + '.json'
    with io.open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    for d1 in data:
        d = defaultdict(list)
        d[''].append(d1[7:].split(',')[0] + ',' + d1[7:].split(',')[1] + ',' + d1[7:].split(',')[2] + ','
                     + d1[7:].split(',')[3] + ',' + d1[7:].split(',')[4] + ',' + d1[7:].split(',')[5]
                     + ',' + d1[7:].split(',')[6] + ',' + d1[7:].split(',')[7][:-4])
        json_str = json.dumps(d)
        with open('D:/dataset2/json/' + '%06d' % new_count + '.json', 'a') as f:
            f.write(json_str + '\n')

def main():
    train_data_dir = "G:/why_workspace/RJB/dataset2/images_enhance/"
    count = 122821
    with open('G:/why_workspace/RJB/dataset2/label4.txt', 'r') as f1:
       data = f1.readlines()
       f1.close()
    for root, dirs, files in os.walk(train_data_dir):
        for file in files:
            img = root + file
            #拉伸
            images_resize(img)
            img_after_brightness, img_after_brightness2, img_after_brightness3, img_after_brightness4 = images_resize(img)
            output_path = 'G:/why_workspace/RJB/dataset2/images_enhance2/' + '%06d' % count + '.png'
            with open('G:/why_workspace/RJB/dataset2/label4.txt', 'a') as f6:
                f6.write('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])
                f6.close()
            count += 1
            cv2.imwrite(output_path, img_after_brightness)
            output_path = 'G:/why_workspace/RJB/dataset2/images_enhance2/' + '%06d' % count + '.png'
            with open('G:/why_workspace/RJB/dataset2/label4.txt', 'a') as f8:
                f8.write('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])
                f8.close()
            count += 1
            cv2.imwrite(output_path, img_after_brightness2)
            output_path = 'G:/why_workspace/RJB/dataset2/images_enhance2/' + '%06d' % count + '.png'
            with open('G:/why_workspace/RJB/dataset2/label4.txt', 'a') as f8:
                f8.write('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])
                f8.close()
            count += 1
            cv2.imwrite(output_path, img_after_brightness3)
            output_path = 'G:/why_workspace/RJB/dataset2/images_enhance2/' + '%06d' % count + '.png'
            with open('G:/why_workspace/RJB/dataset2/label4.txt', 'a') as f8:
                f8.write('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])         
            count += 1
            cv2.imwrite(output_path, img_after_brightness4)  
            #旋转 计算坐标
            for degree in range(-2, 3):
                if degree == 0:
                    continue
                output_path = 'G:/why_workspace/RJB/dataset2/images_enhance/' + '%06d' % count + '.png'
                img_afterdegree, h, w = rotate_img(img, degree)
                #count_rotate(int(file[:6]), count, h, w, degree)
                #print(file[:6] + data[int(file[:6])-1][11:])
                #print( '%06d' % count + '.png ' + data[int(file[:6])-1][11:])
                with open('G:/why_workspace/RJB/dataset2/label4.txt', 'a') as f2:
                    f2.write('%06d' % count + '.png ' + data[int(file[:6])-1][11:])
                    f2.close()
                    #print('%06d' % count + '.png ' + data[int(file[:6])-1][11:])
                count += 1
                #print(output_path)
                cv2.imwrite(output_path, img_afterdegree)
            #wap 计算坐标
            img_after_wap1, img_after_wap2 = wap(img)
            output_path = 'G:/why_workspace/RJB/dataset2/images_enhance_fs/' + '%06d' % count + '.png'
            with open('G:/why_workspace/RJB/dataset2/label3.txt', 'a') as f3:
                f3.write('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])
                #print('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])
            count+=1
            cv2.imwrite(output_path, img_after_wap1)
            output_path = 'G:/why_workspace/RJB/dataset2/images_enhance_fs/' + '%06d' % count + '.png'
            with open('G:/why_workspace/RJB/dataset2/label3.txt', 'a') as f4:
                f4.write('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])
                #print('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])
            count+=1
            cv2.imwrite(output_path, img_after_wap2)
            #亮度 坐标不变
            img_after_brightness, img_after_brightness2, img_after_brightness3, img_after_brightness4 = brightness(img)
            output_path = 'G:/why_workspace/RJB/dataset2/images_enhance/' + '%06d' % count + '.png'
            with open('G:/why_workspace/RJB/dataset2/label3.txt', 'a') as f6:
                f6.write('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])
                #print('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])
            # count_json_brightness(int(file[:6]), count)
            count += 1
            cv2.imwrite(output_path, img_after_brightness2)
            output_path = 'G:/why_workspace/RJB/dataset2/images_enhance/' + '%06d' % count + '.png'
            with open('G:/why_workspace/RJB/dataset2/label3.txt', 'a') as f8:
                f8.write('%06d' % count + '.png ' + data[int(file[:6]) - 1][11:])
            # count_json_brightness(int(file[:6]), count)
            count += 1
            cv2.imwrite(output_path, img_after_brightness4)


if __name__ == '__main__':
    main()
