import sys
import tensorflow as tf
import os
import shutil
from datetime import datetime


def log_run(log_dir, log_to_file=True):
    """
    Bereite das Logging für einen Trainings-Run vor. Die Ausgaben der Konsole
    können dabei in ein Log-File umgeleitet werden.
    :param log_dir: Directory, in welches geloggt wird.
    :param log_to_file: Soll in eine Datei geloggt werden?
    :return: Callback für Tensorboard, Callback für ModelCheckpoints, Verzeichnis zum Log Path
    """
    script_path = sys.argv[0]
    script_name = os.path.basename(script_path)
    log_dir_script = log_dir + script_name + "_" + datetime.now().strftime("%d-%m-%Y_%H%M%S") + "/"
    # Erstelle das Log Verzeichnis
    os.makedirs(log_dir_script)
    # Kopiere das Script, welches ausgeführt wurde in das Log Verzeichnis
    shutil.copy(script_path, log_dir_script + script_name)
    # Leite Ausgabe der Konsole um
    if log_to_file:
        print("Starting Log To File!")
        sys.stdout = open(log_dir_script + script_name + ".log.txt", "w")

    # Erstelle und gebe Callbacks für das Log-Verzeichnis zurück.
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir_script, profile_batch=5), \
           tf.keras.callbacks.ModelCheckpoint(log_dir_script + 'model.mdl_wts.hdf5', save_best_only=True,
                                              monitor='val_loss', mode='min'), log_dir_script
