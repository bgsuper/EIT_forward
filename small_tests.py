import utils.load_mat_customized as lm
from utils.system_mat_fields import *

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


data = lm.loadmat('./data/simple_circle_data_background_nodal')

img = data['img']
fwd_model = img['fwd_model']
impeds = data['impeds']

p = fwd_model_parameters(fwd_model)
d0 = p['n_dims'] + 0
d1 = p['n_dims'] + 1
e = p['n_elem']
(p['NODE'][p['ELEM'][100]])
FF1, FF2, CC1, CC2 = system_mat_fields(fwd_model)


# sv = tf.Variable(sparse)
init = tf.initialize_all_tables()
sess = tf.Session()
sparse =
sess.run(init)
dfc = sess.run( tf.sparse_tensor_to_dense(sparse))