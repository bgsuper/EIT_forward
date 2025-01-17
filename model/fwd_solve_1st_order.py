from model.system_mat_1st_order import *
from model.system_mat_fields import *
import collections

def fwd_solve_1st_order(fwd_model, img):
    params = fwd_model_parameters(fwd_model)
    bdy = params['boundary']
    n_elec = params['n_elec']
    n_nodes = params['n_node']
    n_elem = params['n_elem']
    p = len(fwd_model['stimulation'])

    N2E, QQ = calculate_N2E_QQ(fwd_model, bdy, n_elec, n_nodes, p)
    data= collections.defaultdict()
    E = system_mat_1st_order(fwd_model, img)
    E_ = E[1:, 1:]
    E_inv = tf.matrix_inverse(E_)
    vv = tf.matmul(E_inv, QQ.toarray()[1:])
    vv = tf.concat([tf.zeros([1, n_elec],dtype=tf.float32), vv], axis=0)

    data['inner'] = vv
    data['meas'] = tf.matmul(N2E.toarray(), vv, a_is_sparse=True)

    return data

def meas_from_v_els(v_els, stim):
    n_elec, n_stim = v_els.shape

    v2meas = get_v2meas(n_elec, n_stim, stim)
    vv = np.matmul(v2meas.T, v_els.flatten())

    return vv

def get_v2meas(n_elec, n_stim, stim):
    meas_pat = stim[0]['meas_pattern']
    n_meas = meas_pat.shape[0]
    v2meas = sparse.coo_matrix((n_elec*n_stim, n_stim*n_meas)).tolil()
    for i in range(n_stim):
        meas_pat = stim[i]['meas_pattern']
        n_meas = meas_pat.shape[0]
        v2meas[i*n_elec: (i+1)*n_elec,i*n_meas: (i+1)*n_meas] = meas_pat.T

    return v2meas