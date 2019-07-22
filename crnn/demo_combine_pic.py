import PIL.Image as Image
import os
# 功能:拼接图像


IMAGES_PATH = 'G:/why_workspace/RJB/iam/tmp/images3/'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.png']  # 图片格式
IMAGE_SIZE = 256  # 每张小图片的大小
IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 2  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'G:/why_workspace/RJB/iam/tmp/images4/final.png'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

print(image_names)
# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")


# 定义图像拼接函数

to_image = Image.new('RGB', (IMAGE_COLUMN * 120, IMAGE_ROW * 46))  # 创建一个新图
# 循环遍历，把每张图片按顺序粘贴到对应位置上
for y in range(1, IMAGE_ROW + 1):
    for x in range(1, IMAGE_COLUMN + 1):
        from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
            (120, 46), Image.ANTIALIAS)
        to_image.paste(from_image, ((x - 1) * 120, (y - 1) * 46))
to_image.save(IMAGE_SAVE_PATH)  # 保存新图


