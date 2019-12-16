import tensorflow as tf
import lmbspecialops as sops
import numpy as np

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.contrib.slim import add_arg_scope
from tensorflow.contrib.slim import layers
import sys

from depthmotionnet.helpers import *

def basic_conv_lstm_cell_leakyrelu(inputs,
                         state,
                         num_channels,
                         rate=1,
                         filter_size=5,
                         forget_bias=1.0,
                         stride=1,
                         scope=None,
                         reuse=None):
  """LSTM  with leakyRELU activation, with 2D convolution connctions.
  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.
  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.
  Args:
    inputs: input Tensor, 4D, batch x height x width x channels.
    state: state Tensor, 4D, batch x height x width x channels.
    num_channels: the number of output channels in the layer.
    filter_size: the shape of the each convolution filter.
    forget_bias: the initial value of the forget biases.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and the variables should be reused.
  Returns:
     a tuple of tensors representing output and the new state.
  """
  spatial_size = inputs.get_shape()[1:3]
  if state is None:
    state = init_state(inputs, list(spatial_size) + [2 * num_channels])
  with tf.variable_scope(scope,
                         'BasicConvLstmCell',
                         [inputs, state],
                         reuse=reuse):
    inputs.get_shape().assert_has_rank(4)
    state.get_shape().assert_has_rank(4)
    c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
    inputs_h = tf.concat(axis=3, values=[inputs, h])
    # Parameters of gates are concatenated into one conv for efficiency.
    i_j_f_o = layers.conv2d(inputs_h,
                            4 * num_channels, [filter_size, filter_size],
                            stride=stride,
                            rate=rate,
                            activation_fn=None,
                            scope='Gates')

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=i_j_f_o)

    new_c = c * tf.sigmoid(f + forget_bias) + tf.sigmoid(i) *tf.nn.leaky_relu(j, alpha=0.1)
    new_h = tf.nn.leaky_relu(new_c,alpha=0.1) * tf.sigmoid(o)

    return new_h, tf.concat(axis=3, values=[new_c, new_h])

def conv2d(inputs, num_outputs, kernel_size, data_format, **kwargs):
    """Convolution with 'same' padding"""

    return tf.layers.conv2d(
        inputs=inputs,
        filters=num_outputs,
        kernel_size=kernel_size,
        kernel_initializer=default_weights_initializer(),
        padding='same',
        data_format=data_format,
        **kwargs,
        )


def convrelu(inputs, num_outputs, kernel_size, data_format, **kwargs):
    """Shortcut for a single convolution+relu 
    
    See tf.layers.conv2d for a description of remaining parameters
    """
    return conv2d(inputs, num_outputs, kernel_size, data_format, activation=myLeakyRelu, **kwargs)


def convrelu2(inputs, num_outputs, kernel_size, name, stride, data_format, **kwargs):
    """Shortcut for two convolution+relu with 1D filter kernels 
    
    num_outputs: int or (int,int)
        If num_outputs is a tuple then the first element is the number of
        outputs for the 1d filter in y direction and the second element is
        the final number of outputs.
    """
    if isinstance(num_outputs,(tuple,list)):
        num_outputs_y = num_outputs[0]
        num_outputs_x = num_outputs[1]
    else:
        num_outputs_y = num_outputs
        num_outputs_x = num_outputs

    if isinstance(kernel_size,(tuple,list)):
        kernel_size_y = kernel_size[0]
        kernel_size_x = kernel_size[1]
    else:
        kernel_size_y = kernel_size
        kernel_size_x = kernel_size

    tmp_y = tf.layers.conv2d(
        inputs=inputs,
        filters=num_outputs_y,
        kernel_size=[kernel_size_y,1],
        strides=[stride,1],
        padding='same',
        activation=myLeakyRelu,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name+'y',
        **kwargs,
    )
    return tf.layers.conv2d(
        inputs=tmp_y,
        filters=num_outputs_x,
        kernel_size=[1,kernel_size_x],
        strides=[1,stride],
        padding='same',
        activation=myLeakyRelu,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name+'x',
        **kwargs,
    )


def recursive_median_downsample(inp, iterations):
    """Recursively downsamples the input using a 3x3 median filter"""
    result = []
    for i in range(iterations):
        if not result:
            tmp_inp = inp
        else:
            tmp_inp = result[-1]
        result.append(sops.median3x3_downsample(tmp_inp))
    return tuple(result)
