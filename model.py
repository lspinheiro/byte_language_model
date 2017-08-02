# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 19:26:47 2017

@author: Leonardo
"""

import tensorflow as tf
import numpy as np
import html

from mlstm import MultiplicativeLSTMCell

def batch_pad(xs, nbatch, nsteps):
    xmb = np.zeros((nbatch, nsteps), dtype=np.int32)
    mmb = np.ones((nbatch, nsteps, 1), dtype=np.float32)
    for i, x in enumerate(xs):
        l = len(x)
        npad = nsteps-l
        xmb[i, -l:] = list(x)
        mmb[i, :npad] = 0
    return xmb, mmb

def preprocess(text, front_pad='\n ', end_pad=' '):
    text = html.unescape(text)
    text = text.replace('\n', ' ').strip()
    text = front_pad+text+end_pad
    text = text.encode()
    return text

class CharRNN:
    
    def __init__(self, vocab_size, embed_dim, batch_size, n_steps=128,n_hidden=1024, n_states=2, clip_val=40):
        
        #nsteps?
        
        #M = tf.placeholder(tf.float32, [None, n_steps, 1])
        #S = tf.placeholder(tf.float32, [n_states, None, n_hidden])
        self.input_text, self.targets, self.learning_rate = self.get_inputs(n_steps)
        self.embedding = self.embed_layer(self.input_text, vocab_size, embed_dim)
        
        self.rnn_cell, self.init_state = self.get_init_cell(batch_size, n_hidden)
        self.rnn_output, self.final_state = self.rnn_layer(self.rnn_cell, self.input_embed)
        self.logits, self.out = self.final_fully_connected_layer(self.rnn_output, vocab_size)
        self.cost = self.cost_layer(self.logits, self.targets, batch_size, vocab_size)
        self.train_op = self.optimizer(self.learning_rate, self.cost, clip_val)
        
            
    def get_inputs(self, n_steps):
        
        with tf.variable_scope('inputs'):
            
            input_text = tf.placeholder(tf.int32, [None, n_steps]) #TODO
            targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target')
            learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name='learning_rate')
            
            return input_text, targets, learning_rate
        
    def embed_layer(self, input_text, vocab_size, embed_dim):
        
        with tf.variable_scope('embedding'):
            embedding_lookup_matrix = tf.Variable(tf.random_normal([vocab_size, embed_dim], stddev=0.1), 
                                           name="embedding_lookup_matrix")
            
            embeddings =  tf.nn.embedding_lookup(embedding_lookup_matrix, input_text)
        return embeddings
    
    def get_init_cell(self, batch_size, lstm_size):
        
        cell = tf.contrib.rnn.MultiRNNCell(
                [MultiplicativeLSTMCell(lstm_size, forget_bias=1.0)]
            )
        
        init_state = tf.identity(cell.zero_state(batch_size, dtype=tf.float32), name='initial_state')
        
        return cell, init_state
        
    def rnn_layer(self, rnn_cell, input_embed):
        
        outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=input_embed, dtype=tf.float32)
        final_state = tf.identity(state, name='final_state')
        
        return outputs, final_state
    
    def final_fully_connected_layer(self, rnn_output, vocab_size):
        
        #seq_output = tf.concat(rnn_output, axis=1)
        #x = tf.reshape(seq_output, [-1, lstm_size])
        logits = tf.contrib.layers.fully_connected(rnn_output, vocab_size, activation_fn= None)
        out = tf.nn.softmax(logits, name='predictions')
        
        return logits, out
    
    def cost_layer(self, logits, targets, batch_size, vocab_size):
        
        # Loss function
        y_one_hot = tf.one_hot(targets, vocab_size)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
        loss = tf.reduce_mean(loss)
        
        return loss
    
    def optimizer(self, learn_rate, cost, clip_val):
        
        trainables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainables), clip_val)
        optimizer = tf.train.AdamOptimizer(learn_rate)
        
        train_op = optimizer.apply_gradients(list(zip(grads, trainables)))
        return train_op
            
            

def dynamic_batching(full_batch_sequences, n_steps):
    
    full_batch_sequences = [preprocess(sequence) for sequence in full_batch_sequences] 
    sizes = np.asarray([len(sequence) for sequence in full_batch_sequences])
    
    sorted_idxs = np.argsort(sizes)
    #unsort_idxs = np.argsort(sorted_idxs)
    sorted_full_batch_sequences = [full_batch_sequences[i] for i in sorted_idxs]
    
    max_seq_size = np.max(sizes)
    num_seq_offset = 0
    step_ceil = int(np.ceil(max_seq_size / nsteps) * nsteps)
    
    for step in range(0, step_ceil, nsteps):
        
        start = step
        end = step+nsteps
        batch_subsequences = [x[start:end] for x in sorted_full_batch_sequences]
        num_seq_done = sum([x == b'' for x in batch_subsequences])
        num_seq_offset += num_seq_done
        batch_subsequences = batch_subsequences[num_seq_done:]
        sorted_full_batch_sequences = sorted_full_batch_sequences[num_seq_done:]
        batch_size = len(batch_subsequences)
        
        input_sequences, _ = batch_pad(batch_subsequences, batch_size, n_steps)
        
        yield input_sequences
       