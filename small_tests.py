from model.system_mat_fields import *
import model.system_mat_1st_order as sysmat
from model.fwd_solve_1st_order import *
import numpy as np
import tensorflow as tf
from utils import load_mat_customized as lm

# '''' =========test FC matrix ========='''
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
#
'''' =========test systemat ========='''

data = lm.loadmat('./data/model_sysmat')

# imdl = data['imdl']
fwd_model = data['fwd_model']
img = data['img']
E_tr = data['E']
print(E_tr.shape)
FC = data['FC']
pp = fwd_model_parameters(fwd_model)


# # modify original data for test
# zc = np.zeros([pp['n_elec'], 1])
#
# for i in range(pp['n_elec']):
#     fwd_model['electrode'][i].z_contact = np.random.random(1)[0]
#     zc[i] = fwd_model['electrode'][i].z_contact
#
# img['elem_data'] =np.random.random([pp['n_elem'], 1])

E_0 = sysmat.system_mat_1st_order(fwd_model, img)
E = sysmat.system_mat_1st_order_elec(fwd_model, img)

init = tf.initialize_all_tables()
sess = tf.Session()
sess.run(init)
dfc = sess.run(E)
FCc, FF1, FF2, CC1, CC2 = system_mat_fields(fwd_model, elec_imped=True)
fc1 = tf.sparse_matmul(tf.sparse_tensor_to_dense(FF1, validate_indices=False), tf.sparse_tensor_to_dense(CC1, validate_indices=False), a_is_sparse=True, b_is_sparse=True)
fc2 = tf.sparse_matmul(tf.sparse_tensor_to_dense(FF2, validate_indices=False), tf.sparse_tensor_to_dense(CC2, validate_indices=False), a_is_sparse=True, b_is_sparse=True)

fcc = tf.add(fc1, fc2)
#assert (np.max(np.abs(sess.run(E - E_0)))< 1e-4)
#assert(np.max(np.abs(sess.run(FCc - fcc))) < 1e-6)
#assert(np.max(np.abs(dfc- E_tr) < 1.0e-5)

# print
data = lm.loadmat('./data/model_sysmat')

# imdl = data['imdl']
fwd_model = data['fwd_model']
el = fwd_model['stimulation'][1]
img = data['img']
bdy = fwd_model['boundary']
n_elec = pp['n_elec']
n = pp['n_node']
p =pp['n_stim']
n2e, qq = calculate_N2E_QQ(fwd_model, bdy, n_elec, n, p)
vd = fwd_solve_1st_order(fwd_model, img)
print(sess.run(vd['meas']))



