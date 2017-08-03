# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:50:32 2017

@author: Leonardo Pinheiro
"""

import tensorflow as tf

class CharRNNOpenAI:
    
    def __init__(self, vocab_size, embed_dim, batch_size, n_steps=128,n_hidden=1024, n_states=2, clip_val=40):
        
        #nsteps?
        
        #M = tf.placeholder(tf.float32, [None, n_steps, 1])
        #S = tf.placeholder(tf.float32, [n_states, None, n_hidden])
        self.input_text, self.targets, self.state_var, self.learning_rate = self.get_inputs(n_steps, n_hidden)
        self.embedding = self.embed_layer(self.input_text, vocab_size, embed_dim)
        
        self.rnn_cell, self.init_state = self.get_init_cell(batch_size, n_hidden)
        self.rnn_output, self.final_state = self.rnn_layer(self.rnn_cell, self.embedding)
        self.logits, self.out = self.final_fully_connected_layer(self.rnn_output, vocab_size)
        self.cost = self.cost_layer(self.logits, self.targets, batch_size, vocab_size)
        self.train_op = self.optimizer(self.learning_rate, self.cost, clip_val)
        
            
    def get_inputs(self, n_steps, n_hidden):
        
        with tf.variable_scope('inputs'):
            
            input_text = tf.placeholder(tf.int32, [None, n_steps]) #TODO
            targets = tf.placeholder(dtype=tf.int32, shape=[None, n_steps], name='target')
            state = tf.placeholder(tf.float32, [n_states, None, n_hidden])
            learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name='learning_rate')
            
            return input_text, targets, state, learning_rate
        
    def embed_layer(self, input_text, vocab_size, embed_dim):
        
        with tf.variable_scope('embedding'):
            embedding_lookup_matrix = tf.Variable(tf.random_normal([vocab_size, embed_dim], stddev=0.1), 
                                           name="embedding_lookup_matrix")
            
            embeddings =  tf.nn.embedding_lookup(embedding_lookup_matrix, input_text)
        return embeddings
    
    def mLSTM(self, input_text, state_var, n_hidden, n_steps):
        
        inputs = tf.unstack(input_text, n_steps, 1)
        vocab_size = inputs[0].get_shape()[1].value
        
        
    
    '''
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
    '''
    
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