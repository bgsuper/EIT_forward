from utils.system_mat_fields import system_mat_fields, fwd_model_parameters
import numpy as np
import tensorflow as tf


def model(voltage_inner_init, # 16 or 32 X n_nodes
          voltage_meas,
          voltage_inner_shape,
          n_dim,# n_dim, # dim of node
          sigma_shape, # 1 X N
          FC,
          FC_t,
          FC_shape):

    with tf.variable_scope("update_voltage") as scope:
        voltage_inner = tf.get_variable(name='voltage_inner',
                                        shape=voltage_inner_shape,
                                        initializer=voltage_inner_init,
                                        trainable=True)
        voltage = tf.concat(voltage_inner, voltage_meas, axis=0, name='voltage')

    with tf.variable_scope("update_sigma") as scope:
        sigma = tf.get_variable(name='sigma',
                                shape=sigma_shape,
                                initializer=tf.ones_initializer(sigma_shape, dtype=tf.float32),
                                trainable=True)

        elem_sigma = tf.contrib.kfac.utils.kronecker_product(sigma,
                                                             tf.ones([n_dim, 1], dtype=tf.float32))
        elem_sigma = tf.concat([elem_sigma, tf.ones([FC_shape[0] - n_dim*sigma_shape[1], 1], dtype=tf.float32)],
                               axis=0)

        E_ = tf.matmul(FC_t,
                       tf.multiply(tf.tile(elem_sigma, multiples=[1, FC_shape[1]]), FC),
                       a_is_sparse=True,
                       b_is_sparse=True)

        E = tf.add(tf.transpose(E_), 0.5*E_, name='sys_mat')

        out = tf.matmul(E, voltage, a_is_sparse=True)

    return out


def loss_cal(out, ground_truth, reg, opt):
    # get trainable variables
    with tf.variable_scope("update_sigma") as scope:
        sigma = tf.get_variable('sigma')

    with tf.variable_scope("update_voltage") as scope:
        voltage_inner = tf.get_variable("")

    # def loss
    if opt=='sigma':
        loss = tf.reduce_mean(
            tf.nn.l2_loss(out-ground_truth) + reg*tf.nn.l2_loss(sigma)
        )
    else:
        loss = tf.reduce_mean(
            tf.nn.l2_loss(out - ground_truth) + reg*tf.nn.l2_loss(voltage_inner)
        )
    return loss




# tvars = tf.trainable_variables()
# g_vars = [var for var in tvars if 'g_' in var.name]
# g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)
