import sys
import tensorflow as tf
import os
import shutil
from datetime import datetime


def log_run(log_dir, log_to_file=True):
    script_path = sys.argv[0]
    script_name = os.path.basename(script_path)
    log_dir_script = log_dir + script_name + "_" + datetime.now().strftime("%d-%m-%Y_%H%M%S") + "/"
    os.makedirs(log_dir_script)
    shutil.copy(script_path, log_dir_script + script_name)
    if log_to_file:
        print("Starting Log To File!")
        sys.stdout = open(log_dir_script + script_name + ".log.txt", "w")

    return tf.keras.callbacks.TensorBoard(log_dir=log_dir_script, profile_batch=5), \
           tf.keras.callbacks.ModelCheckpoint(log_dir_script + 'model.mdl_wts.hdf5', save_best_only=True,
                                              monitor='val_loss', mode='min'), log_dir_script
