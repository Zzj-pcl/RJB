import PIL.Image as Image
import os
'''
	功能:拼接卡号，制作出5、6、7、8、19等更长的卡号序列，增加识别鲁棒性。
	方法:先拼接出8、12、16、20位长的卡号序列，再根据需要进行裁剪。
'''

IMAGES_PATH = 'G:/why_workspace/RJB/dataset2/20/'  # 图片集地址
IMAGES_FORMAT = ['.png']  # 图片格式
IMAGE_HEIGHT = 46  # 每张小图片的大小
IMAGE_WIDTH = 120
IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 5  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'G:/why_workspace/RJB/dataset2/19/'  # 图片转换后的地址

# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_WIDTH, IMAGE_ROW * IMAGE_HEIGHT))
    filename = 'G:/why_workspace/RJB/dataset2/label3.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    count = 0
    new_count = 7301
    for root, dirs, files in os.walk(IMAGES_PATH):
        for file in files:
            if len(lines[count][11:]) == 5:
                str1 = '%06d' % new_count + lines[count][6:-1] + lines[count][11:-1] + lines[count][11:-1] + lines[count][11:-1] + lines[count][11:]
                with open(filename, 'a', encoding='utf-8') as f:
                    f.write(str1)
                for y in range(1, IMAGE_ROW + 1):
                    for x in range(1, IMAGE_COLUMN + 1):
                        from_image = Image.open(root + '/' + file).resize(
                            (IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
                        to_image.paste(from_image, ((x - 1) * IMAGE_WIDTH, (y - 1) * IMAGE_HEIGHT))
                to_image.save(IMAGE_SAVE_PATH + '%06d' % new_count + '.png')
                new_count += 1
            count += 1
            print(file)

#定义图像裁剪函数
def crop():
    filename = 'G:/why_workspace/RJB/dataset2/label3.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    count = 7300
    new_count = 7301
    for root, dirs, files in os.walk(IMAGES_PATH):
        for file in files:
            str1 = '%06d' % new_count + lines[count][6:-2] + '\n'
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(str1)
                img = root + '/' + file
                img = Image.open(img)
                img = img.crop((0, 0, 120 * 5 / 20 * 19, 46))
                img.save(IMAGE_SAVE_PATH + '%06d' % new_count + '.png')
            new_count += 1
            count += 1
    print(count - 7300)
    print(new_count - 7301)

#批量重命名文件
def rename():
   count = 2861
   for root, dirs, files in os.walk('G:/why_workspace/RJB/dataset2/images_rename'):
       for file in files:
           src = (root+'/'+file)
           dst = (root+'/'+ '%06d' % count + '.png')
           os.rename(src, dst)
           count += 1

if __name__ == '__main__':
    #image_compose()
    crop()
    # rename()