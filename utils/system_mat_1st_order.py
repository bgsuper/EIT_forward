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
    FC = system_mat_fields(fwd_model)
    FC_shape =FC.shape.as_list()
    lFC = FC_shape[0]
    n_dim = pp['n_dim']
    n_elem = pp['n_elem']

    elem_data = img['elem_data']

    if elem_data.shape[2]:
        pass
    else:
        if n_dim==2:
            idx = np.arange(1, n_elem+1, 2)
            # [idx,idx+1,idx,idx+1]'  [idx,idx,idx+1,idx+1]'
            es_indices = np.vstack((np.hstack((idx,idx+1,idx,idx+1, np.arange(n_elem+1, lFC))).T,
                                    np.hstack((idx,idx,idx+1,idx+1, np.arange(n_elem+1, lFC))).T))

            es_data = np.vstack((elem_data.flatten('F'), np.ones[lFC - n_elem, 1])).astype(np.complex64)

            es_shape = [lFC, lFC]


            ES = tf.SparseTensor(es_indices.T,
                                 es_data,
                                 dense_shape=es_shape)


        if n_dim==3:
            idx = np.arange(1, n_elem+1, 3)

            es_indices = np.vstack(
                (np.hstack((idx, idx+1, idx+2, idx, idx+1, idx+2, idx, idx+1, idx+2, np.arange(n_elem+1, lFC))).T,
                 np.hstack((idx, idx, idx, idx+1, idx+1, idx+1, idx+2, idx+2, idx+2, np.arange(n_elem+1, lFC))).T)
            )

            es_data = np.vstack((elem_data.flatten('F'), np.ones[lFC - n_elem, 1])).astype(np.complex64)

            es_shape = [lFC, lFC]

            ES = tf.SparseTensor(es_indices.T,
                                 es_data,
                                 dense_shape=es_shape)

    FC_t = tf.sparse_transpose(FC)
    E = tf.matmul(tf.matmul(FC_t, ES), FC)
    E = 0.5*tf.sparse_add((tf.sparse_transpose(E), E))

    return E

