import itertools
import os
import pickle
import re
from functools import reduce

import h5py
import numpy as np

# 词表地址
data_path = ''
# 数据地址
data_filenames = {
    # 训练数据地址
    'train': 'BaiduTest/baidu_entity_train.txt',
    # 测试数据地址
    'test': 'BaiduTest/baidu_entity_test.txt',
    # 验证数据地址
    'valid': 'BaiduTest/baidu_entity_dev.txt'
}
vocab_file = os.path.join(data_path, 'vocab.h5')


# 分词
def tokenize(sentence):
    return [s.strip() for s in re.split('(\W+)+', sentence) if s.strip()]


# 转换故事格式
def parse_stories(lines):
    stories = []
    story = []
    for line in lines:
        line = line.strip()
        if not line:
            story = []
        else:
            if ' ' in line:
                _, line = line.split(' ', 1)
            if line:
                if '\t' in line:  # 问题 答案 备选答案
                    q, a, answers = line.split('\t')
                    q = tokenize(q)

                    stories.append((story, q, a))
                else:
                    story.append(tokenize(line))
    return stories


# 获取数据
def get_stories(story_file):
    stories = parse_stories(story_file.readlines())
    flatten = lambda story: reduce(lambda x, y: x + y, story)
    stories = [(flatten(story), q, a) for story, q, a in stories]
    return stories


# From keras.preprocessing: https://github.com/fchollet/keras/blob/master/keras/preprocessing/sequence.py
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # 如果集合为空 empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('删除类型 "%s" 不理解' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('样品的张量 %s 位置序列 %s 与预期的张量不一致 %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('填充类型 "%s" 不理解' % padding)
    return x


# 数据 向量化
def vectorize_stories(data, word2idx, doc_max_len, query_max_len):
    X = []
    Xq = []
    Y = []

    for s, q, a in data:
        x = [word2idx[w] for w in s]
        xq = [word2idx[w] for w in q]
        X.append(x)
        Xq.append(xq)
        Y.append(word2idx[a])

    X = pad_sequences(X, maxlen=doc_max_len)
    Q = pad_sequences(Xq, maxlen=query_max_len)
    return (X, Q, np.array(Y))


def build_vocab():
    if os.path.isfile(vocab_file):
        (word2idx, doc_length, query_length) = pickle.load(open(vocab_file, "rb"))
    else:
        stories = []
        for key, filename in data_filenames.items():
            stories = stories + get_stories(open(os.path.join(data_path, filename), encoding="UTF-8"))

        doc_length = max([len(s) for s, _, _ in stories])
        query_length = max([len(q) for _, q, _ in stories])

        print('文档长度 : {}, 查询长度 : {}'.format(doc_length, query_length))
        vocab = sorted(set(itertools.chain(*(story + q + [answer] for story, q, answer in stories))))
        vocab_size = len(vocab) + 1
        print('词表 长度:', vocab_size)
        word2idx = dict((w, i + 1) for i, w in enumerate(vocab))
        pickle.dump((word2idx, doc_length, query_length), open(vocab_file, "wb"))

    return (word2idx, doc_length, query_length)


def load_data(dataset='train'):
    filename = os.path.join(data_path, data_filenames[dataset])
    # 检查预处理数据并加载它
    if os.path.isfile(filename + '.h5'):
        h5f = h5py.File(filename + '.h5', 'r')
        X = h5f['X'][:100]
        Q = h5f['Q'][:100]
        Y = h5f['Y'][:100]
        h5f.close()
    else:
        stories = get_stories(open(filename, encoding="UTF-8"))

        word2idx, doc_length, query_length = build_vocab()

        X, Q, Y = vectorize_stories(stories, word2idx, doc_length, query_length)
        h5f = h5py.File(filename + '.h5', 'w')
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('Q', data=Q)
        h5f.create_dataset('Y', data=Y)
        h5f.close()
    return X, Q, Y
