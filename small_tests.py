import utils.load_mat_customized as lm
from utils.system_mat_fields import *
import utils.system_mat_1st_order as sysmat
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

#
# data = lm.loadmat('./data/most_simple_model')
#
# imdl = data['imdl']
# fwd_model = imdl['fwd_model']
# FC_true = data['FC']
#
# p = fwd_model_parameters(fwd_model)
# d0 = p['n_dims'] + 0
# d1 = p['n_dims'] + 1
# e = p['n_elem']
# (p['NODE'][p['ELEM'][10]])
# FC, FF1, FF2, CC1, CC2 = system_mat_fields(fwd_model)
#
#
# FF = tf.sparse_add(FF1, FF2)
# CC = tf.sparse_add(CC1, CC2)
#
# init = tf.initialize_all_tables()
# sess = tf.Session()
# sparse = tf.matmul(tf.sparse_tensor_to_dense(FF, validate_indices=False),
#                    tf.sparse_tensor_to_dense(CC, validate_indices=False),
#                    a_is_sparse=True,
#                    b_is_sparse=True)
# sess.run(init)
# dfc = sess.run(sparse)
# true_value = np.hstack((np.array([[0.0,-1.0,1.0,0.0], [-2.0,1.0,1.0,0.0]]), np.zeros([2,37])))/2.0
#
#
# assert (np.max(np.abs(dfc[0:2, :] - true_value))< 1.0e-14)
# assert(np.max(np.abs(dfc - FC_true)) < 1e-7)
#
# print("FC matrix tested!!!")

'''' =========test systemat ========='''


data = lm.loadmat('./data/model_sysmat')

imdl = data['imdl']
fwd_model = imdl['fwd_model']
img = data['img']
E_tr = data['sys_mat']['E']

E = sysmat.system_mat_1st_order(fwd_model, img)

init = tf.initialize_all_tables()
sess = tf.Session()
sess.run(init)
dfc = sess.run(E)
print(dfc)
print(E_tr)