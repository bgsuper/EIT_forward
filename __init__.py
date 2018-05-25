from .utils import show_image
from .model.system_mat_fields import fwd_model_parameters

import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

""" model """
tf.app.flags.DEFINE_string('model_dir', '.model/model_sysmat', """ Select FEM model """)


# Training
tf.app.flags.DEFINE_string('log_dir',"./tmp/model_sysmat", #Training is default on, unless testing or finetuning is set to "True"
                           """ dir to store training ckpt """)
tf.app.flags.DEFINE_integer('max_steps', "400",
                            """ max_steps for training """)

tf.app.flags.DEFINE_string('optimizer', "adagrad",
                            """ Optimizer for training. One of: "adam", "SGD", "momentum", "adagrad". """)