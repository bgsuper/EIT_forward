import tensorflow as tf
from model.system_mat_fields import system_mat_fields, fwd_model_parameters, find_bdy_idx, calculate_N2E_QQ, find_electrode_bdy
from model.fwd_solve_1st_order import fwd_solve_1st_order
import numpy as np
from utils.load_mat_customized import loadmat

FLAGS = tf.app.flags.FLAGS


def init_voltage_sigma(fwd_model, v_meas):
    # calculate voltage with sigma=1
    voltage_inner_init, voltage__meas_cal = fwd_solve_1st_order(fwd_model)
    # using the real v_meas and the v_meas_cal for sigma=1
    #  to scale the voltage and conductivity sigma
    init_scale = cal_init_scale(v_meas, voltage_inner_init)
    voltage_inner_init *= init_scale
    # scale initial conductivity
    sigma_init = 1/init_scale*np.ones([fwd_model['elems'].shape[0],1], dtype=np.float32)
    return voltage_inner_init, sigma_init


def cal_init_scale(v_meas, v_homog):
    """ calculate the scale for variables initialization """
    init_scale = np.dot(v_meas, v_homog)/np.linalg.norm(v_homog)
    return init_scale


def load_model_paramters():
    """ load fem model and parameters associate with it"""
    model_data = loadmat(FLAGS.model_dir)
    fwd_model = model_data['fwd_model']

    return fwd_model_parameters(fwd_model)