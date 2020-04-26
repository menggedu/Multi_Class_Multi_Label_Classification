import tensorflow as tf
from utils.data_loader import build_data
from utils.config import root
import os

def create_look_ahead_mask(size):
    """
    tf.linalg.band_part(
        input,
        num_lower,
        num_upper,
        name=None
    )
    input:输入的张量.
    num_lower:下三角矩阵保留的副对角线数量，从主对角线开始计算，相当于下三角的带宽。取值为负数时，则全部保留。
    num_upper:上三角矩阵保留的副对角线数量，从主对角线开始计算，相当于上三角的带宽。取值为负数时，则全部保留。
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def prepare_data(params):

    X_train, X_test, y_train, y_test, vocab, mlb = build_data(params)
    y_train = tf.constant(y_train, tf.float32)
    y_test = tf.constant(y_test, tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # 将数据集缓存到内存中以加快读取速度。
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(params['BUFFER_SIZE'], reshuffle_each_iteration=True).batch(params['BATCH_SIZE'],
                                                                                                   drop_remainder=True)

    test_dataset = test_dataset.batch(params['BATCH_SIZE'])
    # 流水线技术 重叠训练的预处理和模型训练步骤。当加速器正在执行训练步骤 N 时，CPU 开始准备步骤 N + 1 的数据。这样做可以将步骤时间减少到模型训练与抽取转换数据二者所需的最大时间（而不是二者时间总和）。
    # 没有流水线技术，CPU 和 GPU/TPU 大部分时间将处于闲置状态:
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset

def get_params():
    params = {}
    params['vocab_path'] = os.path.join(root, 'data', 'vocab_new.txt')
    params['data_path'] = os.path.join(root, 'data', 'baidu_95.csv')
    params['vocab_save_dir'] = os.path.join(root, 'data')
    params['vocab_size'] = 50000
    params['padding_size'] = 300
    params['BUFFER_SIZE'] = 3000
    params['BATCH_SIZE'] = 128
    params['train_mode'] = "multi_label"

    return params

def get_train_params():
    train_params = {
        'num_layers': 4,
        'd_model': 128,
        'dff': 512,
        'num_heads': 8,

        'input_vocab_size': 50000,
        'output_dim': 95,
        'dropout_rate': 0.1,
        'maximum_position_encoding': 10000,
        'EPOCHS': 10

    }
    return train_params
