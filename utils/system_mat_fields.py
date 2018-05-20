import utils.load_mat_customized as lm
import numpy as np
from itertools import permutations
import time
from scipy.special import binom, comb
import scipy.sparse.linalg as lig
from scipy.linalg import sqrtm
from scipy import sparse
from numpy.linalg import inv, det
import matplotlib.pylab as plt
import collections
import tensorflow as tf


def find_boundary(simp, dim):
    localface = np.array([[1, 2], [1, 3], [2, 3]]).T
    srf_local = simp[:, (localface.T - 1)]
    srf_local = np.reshape(srf_local.T, [dim, -1])  # dim X 3*E
    srf_local = np.sort(srf_local, axis=0).T
    sort_srl, sort_idx = sortrows(srf_local);

    # Find the ones that are the same
    first_ones = sort_srl[:-1, :]
    next_ones = sort_srl[1:, :]
    same_srl = np.where(np.all(first_ones == next_ones, 1))[0]

    # Assume they are all different. then find the same ones
    diff_srl = np.ones((srf_local.shape[0],), dtype=bool)
    diff_srl[same_srl] = 0
    diff_srl[same_srl + 1] = 0

    srf = sort_srl[diff_srl]
    idx = sort_idx[diff_srl]
    idx = np.ceil(idx / (dim + 1))
    return srf, idx


def sort_boundary(bdy):
    np.sort(bdy, axis=1)
    np.lexsort(bdy)


def sortrows(srf_local):
    sort_idx = np.lexsort(srf_local.T[::-1])
    sort_srl = (srf_local.T)[:, sort_idx]
    sort_srl = sort_srl.T
    return sort_srl, sort_idx


# DONE
def find_electrode_bdy(bdy, vtx, elec_nodes):
    bdy_idx, point = find_bdy_idx(bdy, elec_nodes)
    l_bdy_idx = len(bdy_idx[0])
    l_point = len(point)

    if l_bdy_idx > 0 and l_point == 0:
        bdy_area = np.zeros((1, l_bdy_idx))
        for i in range(l_bdy_idx):
            bdy_nodes = bdy[bdy_idx[0][i], :]
            bdy_area[0, i] = tria_area(vtx[bdy_nodes-1, :])
    elif l_bdy_idx == 0 and l_point > 0:
        dims = bdy.shape[1]
        bdy_area = np.zeros((1, l_point))
        for i in range(l_point):
            ff = np.where(np.any(bdy == point[i], axis=1))
            this_area = 0
            for ffp in ff:
                xyz = vtx[bdy[ffp, :]-1, :]
                this_area = this_area + tria_area(xyz)
            bdy_area[0, i] = bdy_area[i] + this_area / dims;
    else:
        print('can`t model this electrode, with {} CEM and {} point'.format(l_bdy_idx, l_point))

    return bdy_idx, bdy_area


def find_bdy_idx(bdy, elec_nodes):
    bdy_els = np.zeros((bdy.shape[0],))
    elec_nodes = np.unique(elec_nodes)
    for nd in elec_nodes:
        bdy_els = bdy_els + np.any(bdy == nd, axis=1)

    ffb = np.where(bdy_els == bdy.shape[1])
    used_nodes = bdy[ffb, :].reshape(-1)
    unused = np.setdiff1d(elec_nodes, used_nodes)
    return ffb, unused


def tria_area(bdy_pts):
    vectors = np.diff(bdy_pts, axis=0)
    if vectors.shape[0] == 2:
        vectors = np.cross(vectors[0], vectors[1])  # 1d array

    d = bdy_pts.shape[0]
    area = np.sqrt(np.sum(vectors ** 2) / (d - 1))
    return area


# ToDo: TEST!
def calculate_N2E_QQ(fwd_model, bdy, n_elec, n, p):
    stim = fwd_model['stimulation']

    cem_electrodes=0
    N2E = sparse.coo_matrix((n_elec, n+n_elec), dtype=np.float32).tolil()
    QQ = sparse.coo_matrix((n+n_elec, p), dtype=np.float32).tolil()

    for i in range(n_elec):
        try:
            elec_nodes =fwd_model['electrode'][i]['nodes']
        except:
            print('Warning: electrode {} has no nodes'.format(i))

        if len(elec_nodes)==1: # point electrode
            N2E[i, elec_nodes]=1
        elif len(elec_nodes)<1:
            raise ValueError('fwd_model_parameters:electrode','zero length electrode specified')
        else:
            bdy_idx = find_electrode_bdy(bdy, [], elec_nodes)

            if bdy_idx: # CEM electrode
                cem_electrodes += 1
                N2E[i, n+cem_electrodes]=1
            else:
                [bdy_idx, srf_area] = find_electrode_bdy(bdy,
                                                         fwd_model['nodes'],
                                                         elec_nodes)
                N2E[i, elec_nodes]=srf_area/sum(srf_area)

    N2E=N2E[:, :(n+cem_electrodes)]
    QQ = QQ[:(n+cem_electrodes),:]

    for i in range(p):
        src = 0
        try:
            src+=N2E.transpose()*stim[i]['stim_pattern']
        except:
            pass

        QQ[:, i] = src
    return N2E, QQ


# DONE
def fwd_model_parameters(fwd_model):
    param = collections.defaultdict()
    param['NODE'] = fwd_model['nodes']
    param['ELEM'] = fwd_model['elems']
    param['boundary'] = fwd_model['boundary']
    param['n_node'] = param['NODE'].shape[0]
    param['n_dims'] = param['NODE'].shape[1]
    param['n_elec'] = len(fwd_model['electrode'])
    param['n_elem'] = param['ELEM'].shape[0]
    param['n_stim'] = fwd_model['stimulation'].shape[0]
    #     param.NODE = fwd_model['nodes']
    #     param.NODE = fwd_model['nodes']

    return param


def compl_elec_mdl(fwd_model, pp):
    d0 = pp['n_dims']
    FFdata = np.zeros((0, d0))
    FFd_block = sqrtm((np.ones(d0) + np.eye(d0)) / 6 / (d0 - 1))
    FFiidx = np.zeros((0, d0))
    FFjidx = np.zeros((0, d0))
    FFi_block = np.tile(np.arange(d0), [d0, 1])
    CCdata = np.zeros((0, d0))
    CCiidx = np.zeros((0, d0))
    CCjidx = np.zeros((0, d0))

    sidx = d0 * pp['n_elem']
    cidx = (d0 + 1) * pp['n_elem']
    i_cem = 0

    for i in range(pp['n_elec']):
        eleci = fwd_model['electrode'][i]
        zc = eleci.z_contact
        bdy_idx, bdy_area = find_electrode_bdy(pp['boundary'], pp['NODE'], eleci.nodes)
        if not bdy_idx:
            continue

        i_cem += 1
        for j in range(bdy_idx[0].shape[0]):
            bdy_nds = pp['boundary'][bdy_idx[0][j], :]
            FFdata = np.vstack((FFdata, FFd_block * np.sqrt(bdy_area[0][j] / zc)))
            FFiidx = np.vstack((FFiidx, FFi_block.T + sidx))
            FFjidx = np.vstack((FFjidx, FFi_block + cidx))

            CCiidx = np.vstack((CCiidx, FFi_block[0:2, :] + cidx))
            CCjidx = np.vstack((CCjidx, bdy_nds-1, (pp['n_node'] + i_cem) * np.ones((1, d0))-1))
            CCdata = np.vstack((CCdata, np.array([1, -1]).reshape(2, 1) * np.ones((1, d0))))
            sidx = sidx + d0
            cidx = cidx + d0
    return FFdata, FFiidx, FFjidx, CCdata, CCiidx, CCjidx


# Partially Done! need to check numercial precise!!!
def system_mat_fields(fwd_model):
    p = fwd_model_parameters(fwd_model)
    d0 = p['n_dims'] + 0
    d1 = p['n_dims'] + 1
    e = p['n_elem']
    n = p['n_node']
    num_elc = p['n_elec']
    FF_shape = [d0 * e, d1 * e]
    CC_shape = [d1 * e, n]

    FFjidx = np.floor(np.arange(d0 * e).T.reshape([d0 * e, 1]) / d0) * d1 * np.ones((1, d1)) + np.ones(
        (d0 * e, 1)).reshape([d0 * e, 1]) * np.arange(1, d1 + 1)
    FFiidx = np.arange(1, d0 * e + 1).T.reshape([d0 * e, 1]) * np.ones((1, d1))
    FFdata = np.zeros([d0 * e, d1]);
    dfact = (d0 - 1) * d0

    for j in range(1, e + 1):
        a = inv(np.hstack((np.ones((d1, 1)), (p['NODE'][p['ELEM'][j - 1] - 1]))))
        idx = np.arange(d0 * (j - 1) + 1, d0 * j + 1)
        FFdata[np.array(idx - 1), 0:d1] = a[np.arange(1, d1), :] / np.sqrt(dfact * np.abs(det(a)))

    CCdata = np.ones((d1 * e, 1))

    [F2data, F2iidx, F2jidx, C2data, C2iidx, C2jidx] = compl_elec_mdl(fwd_model, p)

    FF1_idx = np.vstack((FFiidx.flatten('F'), FFjidx.flatten('F'))).astype('int') - 1
    CC1_idx = np.vstack((np.arange(1, d1 * e + 1), p['ELEM'].flatten())).astype('int') - 1

    nn_elc = C2data.shape[0]
    
    FF_shape =[ffs +nn_elc for ffs in FF_shape]
    if (C2jidx.shape[0]>0 and C2iidx.shape[0]>0):
        CC_shape =[np.max(C2iidx).astype('int')+1,  np.max(C2jidx).astype('int')+1]

    F2_idx = np.vstack((F2iidx.flatten('F'), F2jidx.flatten('F'))).astype('int')
    C2_idx = np.vstack((C2iidx.flatten('F'), C2jidx.flatten('F'))).astype('int')
    
    FFdata = FFdata.astype(np.float32)
    CCdata = CCdata.astype(np.float32)
    F2data = F2data.astype(np.float32)
    C2data = C2data.astype(np.float32)
    
    FF1 = tf.SparseTensor(FF1_idx.T, FFdata.flatten('F'), dense_shape=FF_shape)
    CC1 = tf.SparseTensor(CC1_idx.T, CCdata.flatten('F'), dense_shape=CC_shape)
    
    FF2 = tf.SparseTensor(F2_idx.T, F2data.flatten('F'), dense_shape=FF_shape)
    CC2 = tf.SparseTensor(C2_idx.T, C2data.flatten('F'), dense_shape=CC_shape)
    FF = tf.sparse_add(FF1, FF2)
    CC = tf.sparse_add(CC1, CC2)

    FC = tf.sparse_matmul(tf.sparse_tensor_to_dense(FF, validate_indices=False),
                       tf.sparse_tensor_to_dense(CC, validate_indices=False),
                       a_is_sparse=True,
                       b_is_sparse=True)


    return FC, FF1, FF2, CC1, CC2
