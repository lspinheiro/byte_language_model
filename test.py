# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:03:12 2017

@author: Leonardo Pinheiro
"""

import tensorflow as tf
import html
import numpy as np

from model import dynamic_batching, preproces, batch_pad

def test():
    # sample data
    demo_text = ['Hello world', 'I am groot!', 'Mr. Strange.']
    
    vocab_size = len(set(' '.join(demo_text)))
    embed_dim = 16
    n_steps = 5
    
    session = tf.InteractiveSession()
    
    ### text input test
    
    for batch in dynamic_batching(demo_text, n_steps):
        input_text = batch
        break
    
    # embedding test
    embedding_lookup_matrix = tf.Variable(tf.random_normal([vocab_size, embed_dim], stddev=0.1), 
                                           name="embedding_lookup_matrix")
            
    embeddings =  tf.nn.embedding_lookup(embedding_lookup_matrix, input_text)
    
    
    tf.global_variables_initializer().run()
    embeddings.eval()
    
    return sample_batch