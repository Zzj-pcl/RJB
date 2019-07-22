import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import config
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def freeze_graph(input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    output_node_names = "dense_3/bias/Adam_1/read"
    saver = tf.train.import_meta_graph(input_checkpoint + '/model_step_199000.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(config.logdir)) #恢复图并得到数据
        for op in sess.graph.get_operations():
            print(op.name, op.values())





def main(_):
    # 输入ckpt模型路径
    input_checkpoint = config.logdir
    # 输出pb模型的路径
    out_pb_path = "./pb2/frozen_model.pb"
    # 调用freeze_graph将ckpt转为pb
    freeze_graph(input_checkpoint, out_pb_path)


if __name__ == "__main__":
    tf.app.run()