import utils.load_mat_customized as lm
from utils.system_mat_fields import system_mat_fields, fwd_model_parameters
import numpy as np
from itertools import permutations
import time
from scipy.special import binom, comb
import scipy.sparse.linalg as lig
from scipy.linalg import sqrtm
from numpy.linalg import inv, det
import matplotlib.pylab as plt
import collections
import tensorflow as tf


def system_mat_1st_order( fwd_model, img):
    pp = fwd_model_parameters(fwd_model)
    FC, _,_,_,_  = system_mat_fields(fwd_model)
    FC_shape =FC.shape.as_list()
    lFC = FC_shape[0]
    n_dim = pp['n_dims']
    dim_n_elem = n_dim*pp['n_elem']

    elem_data = img['elem_data'].reshape(pp['n_elem'], 1)

    if len(elem_data.shape)<3:

        elem_sigma = tf.contrib.kfac.utils.kronecker_product(tf.constant(elem_data, dtype=tf.float32), tf.ones([n_dim, 1], dtype=tf.float32))
        elem_sigma = tf.concat([elem_sigma, tf.ones([lFC-dim_n_elem, 1], dtype=tf.float32)], axis=0)

        es_indices = np.matmul(np.ones([2,1]),np.arange(lFC).reshape([1, lFC]))
        es_shape = [lFC, lFC]

        ES = tf.SparseTensor(es_indices.T, tf.squeeze(elem_sigma), es_shape)

    else:
        if n_dim==2:
            idx = np.arange(1, dim_n_elem+1, 2)
            # [idx,idx+1,idx,idx+1]'  [idx,idx,idx+1,idx+1]'
            es_indices = np.vstack((np.hstack((idx,idx+1,idx,idx+1, np.arange(dim_n_elem+1, lFC))).T,
                                    np.hstack((idx,idx,idx+1,idx+1, np.arange(dim_n_elem+1, lFC))).T))

            es_data = np.vstack((elem_data.flatten('F').reshape(dim_n_elem, 1), np.ones([lFC - dim_n_elem, 1]))).astype(np.complex64)

            es_shape = [lFC, lFC]


            ES = tf.SparseTensor(es_indices.T,
                                 es_data.flatten('F'),
                                 dense_shape=es_shape)


        if n_dim==3:
            idx = np.arange(1, dim_n_elem+1, 3)

            es_indices = np.vstack(
                (np.hstack((idx, idx+1, idx+2, idx, idx+1, idx+2, idx, idx+1, idx+2, np.arange(dim_n_elem+1, lFC))).T,
                 np.hstack((idx, idx, idx, idx+1, idx+1, idx+1, idx+2, idx+2, idx+2, np.arange(dim_n_elem+1, lFC))).T)
            )

            es_data = np.vstack((elem_data.flatten('F').reshape(dim_n_elem, 1), np.ones[lFC - dim_n_elem, 1])).astype(np.complex64)

            es_shape = [lFC, lFC]

            ES = tf.SparseTensor(es_indices.T,
                                 es_data.flatten('F'),
                                 dense_shape=es_shape)

    FC_t = tf.transpose(FC)
    E = tf.matmul(FC_t, tf.sparse_tensor_dense_matmul(ES, FC))
    E = 0.5*tf.add(tf.transpose(E), E)

    return E

