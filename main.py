import os
import pprint

import numpy as np
import tensorflow as tf

import data_helper
import test
import train
from model import AlternatingAttention

flags = tf.flags

flags.DEFINE_integer("embedding_dim", 384, "字符嵌入的维数(默认值:384)")
flags.DEFINE_integer("encoding_dim", 128, "查询/文档双向GRU编码的维数")
flags.DEFINE_integer("num_glimpses", 8, "读取期间的预览迭代次数 (默认: 8)")
flags.DEFINE_float("dropout_keep_prob", 0.8, "退出保留概率(默认值:0.8)")
flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 regularizaion lambda (default: 0.0001)")
flags.DEFINE_float("learning_rate", 1e-3, "AdamOptimizer learning rate (default: 0.001)")
flags.DEFINE_float("learning_rate_decay", 0.8,
                   "在损失不减少的半个周期后，学习率会下降多少(默认值:0.8)")

# Training parameters 训练过程 参数
flags.DEFINE_integer("batch_size", 8, "批量大小(默认:32)")
flags.DEFINE_integer("num_epochs", 12, "训练周期数(默认:12)")
flags.DEFINE_integer("evaluate_every", 300, "在这许多步骤之后，在验证集上评估模型(默认值:300)")

flags.DEFINE_boolean("trace", False, "跟踪(加载较小的数据集)")
flags.DEFINE_string("log_dir", "logs", "将摘要日志写入默认目录的目录(./logs/)")

flags.DEFINE_integer("checkpoint_every", 100, "在这许多步骤之后保存模型(默认值:1000)")
flags.DEFINE_string("ckpt_dir", "ckpts", "检查点目录缺省值 (./ckpts/)")
flags.DEFINE_string("restore_file", None, "检查点来加载")

flags.DEFINE_boolean("evaluate", False, "是否在检查点上运行计算历。必须设置restore_file。")


def main(_):
    flags_title = tf.flags.FLAGS

    # 载入 数据
    X_train, Q_train, Y_train = data_helper.load_data('train')
    X_test, Q_test, Y_test = data_helper.load_data('valid')

    vocab_size = np.max(X_train) + 1

    # 创建日志目录 Create directories
    if not os.path.exists(flags_title.ckpt_dir):
        os.makedirs(flags_title.ckpt_dir)

    flags_title.log_dir = "log"
    if not os.path.exists(flags_title.log_dir):
        os.makedirs(flags_title.log_dir)

    # Train Model 训练模型
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess, tf.device(
            '/gpu:0'):
        model = AlternatingAttention(flags_title.batch_size, vocab_size, flags_title.encoding_dim, flags_title.embedding_dim,
                                     flags_title.num_glimpses, session=sess)

        if flags_title.trace:  # 调试跟踪模型 Trace model for debugging
            train.trace(flags_title, sess, model, (X_train, Q_train, Y_train))
            return

        saver = tf.train.Saver()

        if flags_title.restore_file is not None:
            print('[?] 从检查点加载变量 %s' % flags_title.restore_file)
            saver.restore(sess, flags_title.restore_file)

        # 运行评估 Run evaluation
        if flags_title.evaluate:
            if not flags_title.restore_file:
                print('需要指定要计算的restore_file检查点')
            else:
                test_data = data_helper.load_data('test')
                word2idx, _, _ = data_helper.build_vocab()
                test.run(flags_title, model, test_data, word2idx)
        else:
            train.run(flags_title, sess, model,
                      (X_train, Q_train, Y_train),
                      (X_test, Q_test, Y_test),
                      saver)


if __name__ == '__main__':
    tf.app.run()
