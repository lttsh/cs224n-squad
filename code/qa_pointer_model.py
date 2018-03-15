# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, SelfAttn, BidirectionAttn, BasicAttn

from qa_model import QAModel
logging.basicConfig(level=logging.INFO)


class QAPointerModel(QAModel):
    """Extension of the QA Model that uses Self Attention and BIDAF and an output layer"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        QAModel.__init__(self, FLAGS, id2word, word2id, emb_matrix)

    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """
        print("Building Pointer Model")
        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob, num_layers=self.FLAGS.num_layers, mode=self.FLAGS.rnn_cell)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Use context hidden states to attend to question hidden states

        # BIDAG LAYER
        bidaf_layer = BidirectionAttn(self.keep_prob, self.FLAGS.hidden_size)
        bidaf_output = bidaf_layer.build_graph(
          question_hiddens,
          self.qn_mask,
          context_hiddens,
          self.context_mask)
        # attn_output is shape (batch_size, context_len, hidden_size*6)
        bidaf_output = tf.concat([context_hiddens, bidaf_output], axis=2) # bs, c_l, 8h


        #SELF ATTENTION LAYER
        self_attn_layer = SelfAttn(self.keep_prob, 8 * self.FLAGS.hidden_size, self.FLAGS.selfattn_size)
        self_attn_output = self_attn_layer.build_graph(bidaf_output, self.context_mask) # batch_size, context_len, 8 * hidden_size

        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([bidaf_output, self_attn_output], axis=2) # (batch_size, context_len, hidden_size*16)
        self_attention_encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob, num_layers=self.FLAGS.num_layers, name="AttentionEncoder")
        blended_reps = self_attention_encoder.build_graph(blended_reps, self.context_mask) # batch_size, context_len, hidden_size * 2

        # OUTPUT LAYER

        from tensorflow.python.ops.rnn_cell import DropoutWrapper
        from tensorflow.python.ops import rnn_cell
        from modules import masked_softmax
        input_lens = tf.reduce_sum( self.context_mask, reduction_indices=1) # shape (batch_size)
        output_gru_cell = DropoutWrapper(rnn_cell.GRUCell(2 * self.FLAGS.hidden_size), input_keep_prob=self.keep_prob)
        _, hidden_state = tf.nn.dynamic_rnn(
            output_gru_cell,
            blended_reps, sequence_length=input_lens, dtype=tf.float32) # batch_size, 2*H

        start_attention = BasicAttn(self.keep_prob, 2 * self.FLAGS.hidden_size, 2 * self.FLAGS.hidden_size)
        attn_dist, output = start_attention.build_graph(blended_reps, self.context_mask, tf.expand_dims(hidden_state, axis=1))

        self.probdist_start = tf.reshape(attn_dist, shape=tf.shape(self.context_mask))
        self.logits_start, _ = masked_softmax(self.probdist_start, self.context_mask, 1)
        _, hidden_state = tf.nn.dynamic_rnn(
            output_gru_cell,
            output, initial_state=hidden_state, sequence_length=input_lens, dtype=tf.float32)
        end_attention = BasicAttn(self.keep_prob, 2 * self.FLAGS.hidden_size, 2 * self.FLAGS.hidden_size)
        attn_dist, _ = end_attention.build_graph(blended_reps, self.context_mask, tf.expand_dims(hidden_state, axis=1)) # (batch_size, 1, context_len)

        self.probdist_end = tf.reshape(attn_dist, shape=tf.shape(self.context_mask))
        self.logits_end, _ = masked_softmax(self.probdist_end, self.context_mask, 1)
