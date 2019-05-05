import itertools
import os
import pickle
import re
from functools import reduce

import h5py
import numpy as np

data_path = '西药分词数据/'
pre_path = ""
# 数据地址
data_filenames = {
    'train': '注意力西药执业药师训练集（分词版本）训练.txt',
    'test': '注意力西药执业药师训练集（分词版本）测试.txt',
    'valid': '注意力西药执业药师训练集（分词版本）验证.txt'
}
# 词表文件
vocab_file = os.path.join(data_path, 'vocab.h5')


def tokenize(sentence):
    return [s.strip() for s in re.split('(\W+)+', sentence) if s.strip()]


def parse_stories(lines):
    stories = []
    story = []
    for line in lines:
        line = line.strip()
        if not line:
            story = []
        else:
            _, line = line.split(' ', 1)
            if line:
                if '\t' in line:  # query line
                    q, a = line.split('\t')
                    q = tokenize(q)
                    a = tokenize(a)
                    if story and a and q:
                        stories.append((story, q, a))
                else:
                    if line:
                        story.append(tokenize(line))
    return stories


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

    # 在下面的主循环中，使用第一个非空序列检查的样例形状来检查一致性。
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
            continue  # 发现列表为空 empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('删除类型 "%s" 不被理解' % truncating)

        # 检查“trunc”是否具有预期的张量  check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('样品的张量 %s 位置序列 %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('填充类型 "%s" 不理解' % padding)
    return x


def vectorize_stories(data, word2idx, doc_max_len, query_max_len, answer_max_length):
    X = []
    Xq = []
    Y = []

    for s, q, a in data:
        x = [word2idx[w] for w in s]
        xq = [word2idx[w] for w in q]
        xa = [word2idx[w] for w in a]
        X.append(x)
        Xq.append(xq)
        Y.append(xa)

    X = pad_sequences(X, maxlen=doc_max_len)
    Q = pad_sequences(Xq, maxlen=query_max_len)
    Y = pad_sequences(Y, maxlen=answer_max_length)

    return X, Q, Y


def build_vocab():
    if os.path.isfile(vocab_file):
        (word2idx, doc_length, query_length, answer_length, vocab_size) = pickle.load(open(vocab_file, "rb"))
    else:
        stories = []
        for key, filename in data_filenames.items():
            stories = stories + get_stories(open(os.path.join(data_path, filename)))

        doc_length = max([len(s) for s, _, _ in stories])
        query_length = max([len(q) for _, q, _ in stories])
        answer_length = max([len(a) for _, _, a in stories])

        print('最大置信度信息长度: {}, 最大题目长度: {}, 最大选项长度: {}'.format(doc_length, query_length, answer_length))
        vocab = sorted(set(itertools.chain(*(story + q + answer for story, q, answer in stories))))
        vocab_size = len(vocab) + 1
        print('非重复词汇表的长度:', vocab_size)
        word2idx = dict((w, i + 1) for i, w in enumerate(vocab))
        pickle.dump((word2idx, doc_length, query_length, answer_length, vocab_size), open(vocab_file, "wb"))
    return word2idx, doc_length, query_length, answer_length, vocab_size


def load_data(dataset='train'):
    filename = os.path.join(data_path, data_filenames[dataset])
    # 检查预处理数据并加载它 Check for preprocessed data and load that instead
    if os.path.isfile(filename + '.h5'):
        h5f = h5py.File(filename + '.h5', 'r')
        X = h5f['X'][:]
        Q = h5f['Q'][:]
        Y = h5f['Y'][:]
        h5f.close()
    else:
        stories = get_stories(open(filename))

        word2idx, doc_length, query_length, answer_length, vocab_size = build_vocab()

        X, Q, Y = vectorize_stories(stories, word2idx, doc_length, query_length, answer_length)
        h5f = h5py.File(filename + '.h5', 'w')
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('Q', data=Q)
        h5f.create_dataset('Y', data=Y)
        h5f.close()
    return X, Q, Y
