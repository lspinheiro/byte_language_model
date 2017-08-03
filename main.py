# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:17:39 2017

@author: Leonardo
"""

from model import CharRNN
import tensorflow as tf

if __name__ == '__main__':
    
    train_graph = tf.Graph()
    
    tf.reset_default_graph()
    model = CharRNN(batch_size=256, vocab_size=26, embed_dim=1024)
    with train_graph.as_default():
        
        