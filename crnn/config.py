logdir = './new_log' #checkpoint和过程可视化保存目录
root = 'G:/why_workspace/RJB/dataset2/' #根目录
train_tfrecord = root +'tfrecord/train2.tfrecords' #训练集tfrecord保存路径
valid_tfrecord = root +'tfrecord/valid2.tfrecords' #验证集tfrecord保存路径
alphabet_path = root + 'alphabet.txt' #字典表路径

#索引最前3位Padding值
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
#图片尺寸
image_height = 128
image_max_width = 128*5
label_max_len = 20

#网络参数配置
rnn_units = 512
rnn_layers_num = 3
batch_size = 1
learning_rate = 1e-4
l2_reg = 1e-5
decay_steps = 4000
total_steps = 99999999
show_step = 100
val_step = 1000
simple_step = 999999
val_data = 20
eval_times = 10
