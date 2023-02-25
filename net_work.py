from __future__ import absolute_import

import os,sys,codecs
import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)
import tensorflow as tf

sys.path.append("model_utils/")
from basic import linear_layer_act, linear_layer, cosine
from lstm import blstm
import modeling
from cnn import cnn_1d
from crf import crf_layer
from cross_attention import local_inference, local_inference_simple

FLAGS = tf.app.flags.FLAGS

def cnn_multi_kernel(inputs, hidden_size_cnn, keep_prob, layer="layer_1", kernels_size=[]):
    num_kernel = len(kernels_size)
    
    all_cnn_out = []
    for idx, _kernel_size in enumerate(kernels_size):
        all_cnn_out.append(cnn_1d(inputs, hidden_size=hidden_size_cnn/num_kernel,
                                    kernel_size=_kernel_size, stride_size=1,
                                  is_pooling=False, var_scope=layer+"cnn_"+str(_kernel_size),
                                  keep_prob=keep_prob
                                 ))
    cnn_out = tf.concat(all_cnn_out, axis=-1)
    return cnn_out


def cnn_crf(is_training, inputs_idsA, inputs_maskA, label_ids, sequence_length, num_labels, keep_prob=0.5, var_scope="cnn_crf"):
    hidden_size_cnn=FLAGS.hidden_size_cnn
    embedding_size = FLAGS.embedding_size
    vocab_size = FLAGS.vocab_size
    batch_size = tf.shape(inputs_idsA)[0]
    
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # 随机初始化
        embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, inputs_idsA)
        tf.logging.info("shape of inputsA: {}".format(inputs))
        
        if is_training:
            inputs = tf.nn.dropout(inputs, 0.9)
        
        cnn_1 = cnn_multi_kernel(inputs, hidden_size_cnn, keep_prob, layer=var_scope+"_cnn_1", kernels_size=[1,3,5])
        if is_training:
            cnn_1 = tf.nn.dropout(cnn_1, keep_prob)
        
        cnn_2 = cnn_multi_kernel(cnn_1, hidden_size_cnn, keep_prob, layer=var_scope+"_cnn_2", kernels_size=[3,5,7])
        if is_training:
            cnn_2 = tf.nn.dropout(cnn_2, keep_prob)
        
        cnn_3 = cnn_multi_kernel(cnn_2, hidden_size_cnn, keep_prob, layer=var_scope+"_cnn_3", kernels_size=[1,3,5])
        
        linear_out = tf.reshape(cnn_3, [-1, hidden_size_cnn])
        linear_out = linear_layer_act(linear_out, hidden_size_cnn, "linear_act_1", 0.02)
        linear_out = linear_layer(linear_out, num_labels, "linear_out", 0.02)
        
        crf_out = crf_layer(linear_out, label_ids, batch_size, sequence_length, num_labels, FLAGS.max_seq_length, "crf")
        return crf_out
