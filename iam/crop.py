# coding=utf-8
import numpy as np
import cv2
from PIL import Image
import math

# 功能：对图片进行任意四边形裁剪，并保存

#输入：图片路径，需要裁剪的四点坐标，按照左上，右上，右下，左下的顺序。
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
    masked = cv2.bitwise_and(image, image, mask=mask)
    new_x1 = x1
    if x4 < new_x1:
        new_x1=x4
    new_x2 = x2
    if x4 > new_x2:
        new_x2 = x4
    new_y1 = y1
    if y2 < new_y1:
        new_y1 = y2
    new_y2 = y3
    if y4 > new_y2:
        new_y2 = y2
    print('new_x1', new_x1)
    print('new_x2', new_x2)
    print('new_y1', new_y1)
    print('new_y2', new_y2)
    new_masked = masked[new_y1:new_y2, new_x1:new_x2]
    max_h = new_y2 - new_y1
    max_w = new_x2 - new_x1
    print(max_h)
    print(max_w)
    array = np.zeros((max_h, max_w, 4), np.uint8)
    array[:, :, 0:3] = new_masked
    array[:, :, 3] = 0
    array[:, :, 3][np.where(array[:, :, 0] > 2)] = 255
    array[:, :, 3][np.where(array[:, :, 1] > 2)] = 255
    array[:, :, 3][np.where(array[:, :, 2] > 2)] = 255
    image_1 = Image.fromarray(array)
    image_1.save("./222.png", "PNG")

if __name__ == '__main__':
    img_path = './tmp/0_0_check.png'
    x1 = 153
    y1 = 351
    x2 = 361
    y2 = 356
    x3 = 359
    y3 = 426
    x4 = 151
    y4 = 421
    crop_rec(img_path, x1, y1, x2, y2, x3, y3, x4, y4)