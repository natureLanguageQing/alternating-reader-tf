import numpy as np
import tensorflow as tf


def orthogonal_initializer(scale=1.1):
    def _initializer(shape):
        '''
        从keras https://github.com/fchollet/keras/blob/master/keras/initializations.py
        from keras https://github.com/fchollet/keras/blob/master/keras/initializations.py
        '''
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape 选一个张量正确的
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return _initializer


def length(sequence):
    return tf.reduce_sum(tf.sign(tf.abs(sequence)), 1)


class AlternatingAttention(object):
    """
    迭代交替注意机制网络
    Iterative Alternating Attention Network"""

    def __init__(self, batch_size, vocab_size, encoding_size, embedding_size,
                 num_glimpses=8,
                 grad_norm_clip=5.,
                 l2_reg_coef=1e-4,
                 session=tf.Session(),
                 name='AlternatingAttention'):
        """
        创建一个迭代交替注意网络，如https://arxiv.org/abs/1606.02245所述
        Creates an iterative alternating attention network as described in https://arxiv.org/abs/1606.02245
        """
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._encode_size = encoding_size
        self._infer_size = 4 * encoding_size
        self._embedding_size = embedding_size
        self._num_glimpses = num_glimpses
        self._sess = session
        self._name = name

        self._build_placeholders()
        self._build_variables()

        # Regularization 正则化
        tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(l2_reg_coef), [self._embeddings])

        # Answer probability 答案的概率
        doc_attentions = self._inference(self._docs, self._queries)
        nans = tf.reduce_sum(tf.cast(tf.is_nan(doc_attentions), dtype=float))

        self._doc_attentions = doc_attentions
        ans_mask = tf.cast(tf.equal(tf.expand_dims(self._answers, -1), self._docs), dtype=float)
        P_a = tf.reduce_sum(ans_mask * doc_attentions, 1)
        loss_op = -tf.reduce_mean(tf.log(P_a + tf.constant(0.00001)))
        self._loss_op = loss_op

        # Optimizer and gradients 优化和梯度
        with tf.name_scope("optimizer"):
            self._opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            grads_and_vars = self._opt.compute_gradients(loss_op)
            capped_grads_and_vars = [(tf.clip_by_norm(g, grad_norm_clip), v) for g, v in grads_and_vars]
            self._train_op = self._opt.apply_gradients(capped_grads_and_vars, global_step=self._global_step)

        tf.summary.scalar('loss', self._loss_op)
        tf.summary.scalar('learning_rate', self._learning_rate)
        tf.summary.histogram('answer_probability', P_a)
        self._summary_op = tf.summary.merge_all()

        self._sess.run(tf.global_variables_initializer())

    def _build_placeholders(self):
        """
        为模型的输入添加tensorflow占位符:文档、查询、答案。
        keep_prob和learning_rate是我们在训练时可能需要调整的超参数。
        Adds tensorflow placeholders for inputs to the model: documents, queries, answers.
        keep_prob and learning_rate are hyperparameters that we might like to adjust while training.
        """
        self._docs = tf.placeholder(tf.int32, [None, None], name="docs")
        self._queries = tf.placeholder(tf.int32, [None, None], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answers")

        self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self._learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    def _build_variables(self):
        with tf.variable_scope("variables",
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.22, dtype=tf.float32)):
            self._embeddings = tf.get_variable("embeddings", [self._vocab_size, self._embedding_size], dtype=tf.float32)
            self._A_q = tf.get_variable("A_q", [2 * self._encode_size, self._infer_size], dtype=tf.float32)
            self._a_q = tf.get_variable("a_q", [2 * self._encode_size, 1], dtype=tf.float32)

            self._A_d = tf.get_variable("A_d", [2 * self._encode_size, self._infer_size + 2 * self._encode_size],
                                        dtype=tf.float32)
            self._a_d = tf.get_variable("a_d", [2 * self._encode_size, 1], dtype=tf.float32)

            self._g_q = tf.get_variable("g_q", [self._infer_size + 6 * self._encode_size, 2 * self._encode_size])
            self._g_d = tf.get_variable("g_d", [self._infer_size + 6 * self._encode_size, 2 * self._encode_size])

            self._global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                                dtype=tf.int32, trainable=False)

    def _embed(self, sequence):
        """
        为序列中的每个单词执行嵌入查找
        performs embedding lookups for every word in the sequence
        """
        with tf.variable_scope('embed'):
            embedded = tf.nn.embedding_lookup(self._embeddings, sequence)
            return embedded

    def _bidirectional_encode(self, sequence, seq_lens, size):
        """
        用两个gru编码序列，一个向前，一个向后，并返回连接
        Encodes sequence with two GRUs, one forward, one backward, and returns the concatenation
        """
        with tf.name_scope('encode'):
            gru_cell = tf.keras.layers.GRUCell(size)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cell, gru_cell, sequence, sequence_length=seq_lens,
                dtype=tf.float32, swap_memory=True)
            encoded = tf.concat(outputs, 2)
            return encoded

    def _glimpse(self, weights, bias, encodings, inputs):
        """
        计算程序浏览编码。的双线性乘积计算注意权重
        编码、权重矩阵和输入。
        返回注意权重和计算的一瞥
        Computes glimpse over an encoding. Attention weights are computed based on the bilinear product of
        the encodings, weight matrix, and inputs.

        Returns attention weights and computed glimpse
        """
        weights = tf.nn.dropout(weights, self._keep_prob)
        inputs = tf.nn.dropout(inputs, self._keep_prob)
        attention = tf.transpose(tf.matmul(weights, tf.transpose(inputs)) + bias)
        attention = tf.matmul(encodings, tf.expand_dims(attention, -1))
        attention = tf.nn.softmax(tf.squeeze(attention, -1))
        return attention, tf.reduce_sum(tf.expand_dims(attention, -1) * encodings, 1)

    def _inference(self, docs, queries):
        """
        计算 文章 注意力矩阵 给 文章 一个批处理 和 问题 一个批处理
        Computes document attentions given a document batch and query batch.
        """
        with tf.name_scope("inference"):
            # Compute document lengths / query lengths for batch 计算文章长度、问题长度 为了批处理
            doc_lens = length(docs)
            query_lens = length(queries)
            batch_size = tf.shape(docs)[0]

            with tf.variable_scope('encode'):
                # Encode Document / Query 编码（矢量化） 文章 编码（矢量化） 问题
                with tf.variable_scope('docs'), tf.device('/gpu:0'):
                    encoded_docs = tf.nn.dropout(self._embed(docs), self._keep_prob)
                    encoded_docs = self._bidirectional_encode(encoded_docs, doc_lens, self._encode_size)
                with tf.variable_scope('queries'), tf.device('/gpu:1'):
                    encoded_queries = tf.nn.dropout(self._embed(queries), self._keep_prob)
                    encoded_queries = self._bidirectional_encode(encoded_queries, query_lens, self._encode_size)

            with tf.variable_scope('attend') as scope:
                infer_gru = tf.nn.rnn_cell.GRUCell(self._infer_size)
                infer_state = infer_gru.zero_state(batch_size, tf.float32)
                for iter_step in range(self._num_glimpses):
                    if iter_step > 0:
                        scope.reuse_variables()

                    # Glimpse query and document 查询和文档
                    with tf.device('/gpu:0'):
                        q_attention, q_glimpse = self._glimpse(self._A_q, self._a_q, encoded_queries, infer_state)
                        tf.add_to_collection('query_attentions', q_attention)
                    with tf.device('/gpu:1'):
                        d_attention, d_glimpse = self._glimpse(self._A_d, self._a_d, encoded_docs,
                                                               tf.concat([infer_state, q_glimpse], 1))
                        tf.add_to_collection('doc_attentions', d_attention)
                    # Search Gates 搜索门

                    gate_concat = tf.concat([infer_state, q_glimpse, d_glimpse, q_glimpse * d_glimpse], 1)

                    r_d = tf.sigmoid(tf.matmul(gate_concat, self._g_d))
                    r_d = tf.nn.dropout(r_d, self._keep_prob)
                    r_q = tf.sigmoid(tf.matmul(gate_concat, self._g_q))
                    r_q = tf.nn.dropout(r_q, self._keep_prob)

                    combined_gated_glimpse = tf.concat([r_q * q_glimpse, r_d * d_glimpse], 1)
                    _, infer_state = infer_gru(combined_gated_glimpse, infer_state)

            return tf.cast(tf.sign(tf.abs(docs)), dtype=float) * d_attention

    def batch_fit(self, docs, queries, answers, learning_rate=1e-3, run_options=None, run_metadata=None):
        """
        执行批处理训练迭代
        Perform a batch training iteration
        """
        feed_dict = {
            self._docs: docs,
            self._queries: queries,
            self._answers: answers,
            self._keep_prob: 0.8,
            self._learning_rate: learning_rate
        }

        loss, summary, _, step, attentions = self._sess.run(
            [self._loss_op, self._summary_op, self._train_op, self._global_step, self._doc_attentions],
            feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

        return loss, summary, step, attentions

    def get_attentions(self, docs, queries, answers):
        """
        获取每个交替迭代的注意力分布。
        Gets the attention distributions for each alternating iteration.
        """
        feed_dict = {
            self._docs: docs,
            self._queries: queries,
            self._answers: answers,
            self._keep_prob: 1.,
            self._learning_rate: 0.
        }
        d_a, q_a = self._sess.run([
            tf.get_collection('doc_attentions'),
            tf.get_collection('query_attentions')
        ], feed_dict=feed_dict)

        return np.asarray(d_a), np.asarray(q_a)

    def batch_predict(self, docs, queries, answers):
        """
        执行批处理的预测。计算批量预测的准确性。
        Perform batch prediction. Computes accuracy of batch predictions.
        """
        feed_dict = {
            self._docs: docs,
            self._queries: queries,
            self._answers: answers,
            self._keep_prob: 1.,
            self._learning_rate: 0.
        }
        loss, summary, attentions = self._sess.run(
            [self._loss_op, self._summary_op, self._doc_attentions],
            feed_dict=feed_dict)

        return loss, summary, attentions
