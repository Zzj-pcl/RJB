src_root_dir = 'G:/why_workspace/RJB/dataset/' #数据集原路径
dst_root_dir = 'G:/why_workspace/RJB/tfrecord/' #tfrecord存储路径
gt_dir = 'G:/why_workspace/RJB/json/' #json文件路径
img_dir = 'G:/why_workspace/RJB/dataset/' #图片路径
train_save_path = dst_root_dir + 'train.tfrecords' # 训练集集保存路径
valid_save_path = dst_root_dir + 'valid.tfrecords' # 验证集集保存路径

# Padding label
PAD_ID = 0
GO_ID = 1
EOS_ID = 2

# 图片尺寸
max_len = 256
max_height = 512
max_width = 512
mask_height = 128
mask_width = 128
line_min_height = 4 # in mask map
line_min_width = 4
# 网络参数
display = 100
total_steps = 6400000
early_stop_count = 24
warmup_steps = 4000
gamma_offset = 5
gamma_theta = 5
gamma_local = 1
shuffle_threads = 2
batch_size = 8
hidden_units = 256
num_heads = 4
sub_rate = 0
dropout_rate = 0
l2_reg = 1e-5
learning_rate = 1e-5
num_encoder_blocks = 2
num_decoder_blocks = 2

# 数据集保存路径
train_tfrecords = [train_save_path]
valid_tfrecords = [valid_save_path]
test_tfrecords = []

# 数据集样本数情况
num_train_samples = 75000
num_valid_samples = 3750
num_total_samples = 78750


logdir = './logdir'


