from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class AlexNet(object):
    def __init__(self, is_training, name, log=False):
        self._is_training = is_training
        self.name = name
        self.num_classes = 2
        self.log = log

    
    def inference(self, inputs):
        with tf.variable_scope(self.name):
            L1 = self._cbrp(inputs, filters=16, kernel_size=[3,3], name='L1')
            L2 = self._cbrp(L1, filters=32, kernel_size=[3,3], name='L2')
            L3 = self._cbr(L2, filters=64, kernel_size=[3,3], name='L3')
            L4 = self._cbr(L3, filters=64, kernel_size=[3,3], name='L4')
            L5 = self._cbr(L4, filters=32, kernel_size=[3,3], name='L5')
            
            F1 = tf.layers.dense(tf.layers.flatten(L5), units=512, activation=tf.nn.relu, name='F1')
            F2 = tf.layers.dense(F1, units=512, activation=tf.nn.relu, name='F2')
            output = tf.layers.dense(F2, units=self.num_classes, name='output')
            
            if self.log:
                print(f'L1 shape is {L1.get_shape()}')
                print(f'L2 shape is {L2.get_shape()}')
                print(f'L3 shape is {L3.get_shape()}')
                print(f'L4 shape is {L4.get_shape()}')
                print(f'L5 shape is {L5.get_shape()}')
                print(f'F1 shape is {F1.get_shape()}')
                print(f'F2 shape is {F2.get_shape()}')
                print(f'prediction_{self.name} shape is {output.get_shape()}')
            
            return output
    
    
    def _cbrp(self, inputs, filters, kernel_size, name=None):
        with tf.variable_scope(name+'_cbrp') as name_scope:
            conv = tf.layers.conv2d(inputs=inputs,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    padding='same',
                                    use_bias=False,
                                    name='conv')
            
            bn = tf.layers.batch_normalization(inputs=conv,
                                               training=self._is_training,
                                               name='bn')
            
            relu = tf.nn.relu(bn, name='relu')
            
            pool = tf.layers.max_pooling2d(relu,
                                           pool_size=[2,2],
                                           strides=[2,2],
                                           padding='same',
                                           name='pool')
            
            tf.logging.info(f'image after unit {name_scope}: {pool.get_shape()}')
            return pool
        
        
    def _cbr(self, inputs, filters, kernel_size, name=None):
        with tf.variable_scope(name+'_cbr') as name_scope:
            conv = tf.layers.conv2d(inputs=inputs,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    padding='same',
                                    use_bias=False,
                                    name='conv')
            
            bn = tf.layers.batch_normalization(inputs=conv,
                                               training=self._is_training,
                                               name='bn')
            
            relu = tf.nn.relu(bn, name='relu')
            
            tf.logging.info(f'image after unit {name_scope}: {relu.get_shape()}')
            return relu