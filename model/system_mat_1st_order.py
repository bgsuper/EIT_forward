from model.system_mat_fields import system_mat_fields, fwd_model_parameters, find_electrode_bdy
import numpy as np
import tensorflow as tf


def system_mat_1st_order( fwd_model, img=None):
    pp = fwd_model_parameters(fwd_model)
    FC, _,_,_,_  = system_mat_fields(fwd_model, elec_imped=True)
    FC_shape =FC.shape.as_list()
    lFC = FC_shape[0]
    n_dim = pp['n_dims']
    dim_n_elem = n_dim*pp['n_elem']

    try:
        elem_data = img['elem_data'].reshape(pp['n_elem'], 1)
    except KeyError:
        elem_data = np.ones([pp['n_elem'], 1], dtype=np.float32)

    if len(elem_data.shape)<3:

        elem_sigma = tf.contrib.kfac.utils.kronecker_product(tf.constant(elem_data, dtype=tf.float32), tf.ones([n_dim, 1], dtype=tf.float32))
        elem_sigma = tf.concat([elem_sigma, tf.ones([lFC-dim_n_elem, 1], dtype=tf.float32)], axis=0)

        es_indices = tf.cast(tf.matmul(tf.ones([2,1], dtype=tf.int32), tf.reshape(tf.range(lFC), [1, lFC])), tf.int64)
        es_shape = [lFC, lFC]

        ES = tf.SparseTensor(tf.transpose(es_indices), tf.squeeze(elem_sigma), es_shape)
    else:
        # ToDo: need to test for future use
        if n_dim==2:
            idx = np.arange(1, dim_n_elem+1, 2)
            # [idx,idx+1,idx,idx+1]'  [idx,idx,idx+1,idx+1]'
            es_indices = tf.concate((tf.concat((idx,idx+1,idx,idx+1, np.arange(dim_n_elem+1, lFC)), axis=1).T,
                                     tf.concat((idx,idx,idx+1,idx+1, np.arange(dim_n_elem+1, lFC)), axis=1).T),
                                    axis=0)

            es_data = tf.concat((elem_data.flatten('F').reshape(dim_n_elem, 1), np.ones([lFC - dim_n_elem, 1], dtype=np.complex64)), axis=0)

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

def system_mat_1st_order_elec( fwd_model, img):
    pp = fwd_model_parameters(fwd_model)
    FC, FF1, FF2, CC1, CC2 = system_mat_fields(fwd_model, elec_imped=False)
    FC_shape =FC.shape.as_list()
    lFC = FC_shape[0]
    n_dim = pp['n_dims']
    dim_n_elem = n_dim*pp['n_elem']

    elem_data = img['elem_data'].reshape(pp['n_elem'], 1)


    if len(elem_data.shape)<3:
        zc_ = np.zeros([pp['n_elec'], 1])
        zc = np.zeros([0, 1])
        for i in range(pp['n_elec']):
            eleci = fwd_model['electrode'][i]
            zc_[i] = eleci.z_contact
            bdy_idx, bdy_area = find_electrode_bdy(pp['boundary'], pp['nodes'], eleci.nodes)
            zc = np.vstack((zc, np.ones([bdy_idx[0].shape[0]*n_dim, 1], dtype=np.float32)*(1.0/zc_[i])))

        # zc = np.tile(zc, [n_dim, 1])
        elem_sigma = tf.contrib.kfac.utils.kronecker_product(tf.constant(elem_data, dtype=tf.float32), tf.ones([n_dim, 1], dtype=tf.float32))
        #elem_sigma = tf.concat([elem_sigma, tf.ones([lFC-dim_n_elem, 1], dtype=tf.float32)], axis=0)
        elem_sigma = tf.concat([elem_sigma, zc], axis=0)


        es_indices = tf.cast(tf.matmul(tf.ones([2,1], dtype=tf.int32), tf.reshape(tf.range(lFC), [1, lFC])), tf.int64)
        es_shape = [lFC, lFC]

        ES = tf.SparseTensor(tf.transpose(es_indices), tf.squeeze(elem_sigma), es_shape)
    else:
        # ToDo: need to test for complex conductivity
        if n_dim==2:
            idx = np.arange(1, dim_n_elem+1, 2)
            # [idx,idx+1,idx,idx+1]'  [idx,idx,idx+1,idx+1]'
            es_indices = tf.concate((tf.concat((idx,idx+1,idx,idx+1, np.arange(dim_n_elem+1, lFC)), axis=1).T,
                                     tf.concat((idx,idx,idx+1,idx+1, np.arange(dim_n_elem+1, lFC)), axis=1).T),
                                    axis=0)

            es_data = tf.concat((elem_data.flatten('F').reshape(dim_n_elem, 1), np.ones([lFC - dim_n_elem, 1], dtype=np.complex64)), axis=0)

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