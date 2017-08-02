# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import RNNCell

def orthogonal_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

class MultiplicativeLSTMCell(RNNCell):
    """Multiplicative LSTM.
       Ben Krause, Liang Lu, Iain Murray, and Steve Renals,
       "Multiplicative LSTM for sequence modelling, "
       in Workshop Track of ICLA 2017,
       https://openreview.net/forum?id=SJCS5rXFl&noteId=SJCS5rXFl
    """

    def __init__(self, num_units,
                 cell_clip=None,
                 initializer=orthogonal_initializer(),
                 forget_bias=1.0,
                 activation=tf.tanh):
        """Initialize the parameters for an LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          use_peepholes: bool, set True to enable diagonal/peephole
            connections.
          cell_clip: (optional) A float value, if provided the cell state
            is clipped by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight
            matrices.
          forget_bias: Biases of the forget gate are initialized by default
            to 1 in order to reduce the scale of forgetting at the beginning of
            the training.
          activation: Activation function of the inner states.
        """
        self.num_units = num_units
        self.cell_clip = cell_clip
        self.initializer = initializer
        self.forget_bias = forget_bias
        self.activation = activation
        self._state_size = tf.contrib.rnn.LSTMStateTuple(num_units, num_units)
        self._output_size = num_units


    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):

        
        (c_prev, h_prev) = state

        #dtype = inputs.dtype
        #input_size = inputs.get_shape().with_rank(2)[1]

        with tf.variable_scope(scope or type(self).__name__):
            
            # Linear calculates eq.18 components
            with tf.variable_scope("Multipli_Weight"):
                concat = _linear([inputs, h_prev], 2 * self.num_units, True)
                
            Wmx_xt, Wmh_hprev = tf.split(concat, 2, 1)
            m = Wmx_xt * Wmh_hprev  # equation (18)

            with tf.variable_scope("LSTM_Weight"):
                lstm_matrix = _linear([inputs, m], 4 * self.num_units, True)
            i, h_hat, f, o = tf.split(lstm_matrix, 4, 1)
            
            c = c_prev * tf.sigmoid(f + self.forget_bias) + \
                tf.sigmoid(i) * h_hat

            h = self.activation(c * tf.sigmoid(o))


            new_state = tf.contrib.rnn.LSTMStateTuple(c, h)
            

            return h, new_state


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        
        res = tf.matmul(tf.concat(args,1), matrix)
        
        if not bias:
            return res
        
        bias_term = tf.get_variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term
