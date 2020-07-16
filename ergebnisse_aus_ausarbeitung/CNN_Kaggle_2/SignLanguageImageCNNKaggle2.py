"""
CNN Test auf dem ASL-Alphabet Datensatz beschrieben in der Ausarbeitung.
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization
from sign_language_image.datasets.kaggle_2.DatasetParser import get_train_and_val, get_test, class_names
from model_logging.LogRun import log_run
from Util import evaluate
import numpy
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Batchgröße
batch_size = 100
# Bildgröße
image_size = (200, 200)

train_ds, val_ds = get_train_and_val(batch_size, image_size_param=image_size)
test_ds = get_test(image_size_param=image_size)

# Anzahl der Klassen aus den Namen der Klassen extrahieren
num_of_classes = len(class_names)
input_shape = (image_size[0], image_size[1], 1)

# Modell
model = Sequential()
# Steigende Filterzahl, absteigende Filtergröße
model.add(Conv2D(32, (7, 7), input_shape=input_shape, strides=(2, 2)))
model.add(Conv2D(64, (5, 5), strides=(2, 2)))
model.add(Conv2D(128, (5, 5), strides=(2, 2)))
model.add(Conv2D(128, (3, 3), strides=(2, 2)))
model.add(Conv2D(128, (3, 3), strides=(2, 2)))
# Ausgaben der Convolutional Layer flatten
model.add(Flatten())
model.add(Dense(1024, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(num_of_classes, activation='softmax'))

# Modell kompilieren mit Kreuzentropie und Adam Optimizer.
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.00005),
              metrics=['accuracy'])

# Logge den Run. Erhalte Callbacks und RunPath.
tensorboard_callback, checkpoint, run_path = log_run("/home/sebastian/PycharmProject/SignLanguageRecognition/logs/")
model.summary()

# Early Stopping Callback.
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Fitte das Modell.
model.fit(train_ds,
          epochs=100,
          verbose=1,
          validation_data=val_ds,
          callbacks=[tensorboard_callback, checkpoint, es])
# Lade das beste Modell
model = load_model(run_path + "model.mdl_wts.hdf5")
# Score auf den Testdaten
score = model.evaluate(test_ds, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Entnehme aus dem Test Datensatz, zur Anwendung mit sklearn
test_data_labels = []
for data, label in test_ds.take(-1):
    test_data_labels.extend(label)

# Prediction für den Test Datensatz
pred = model.predict(test_ds, verbose=1)
# Evaluiere die Ergebnisse vom Testdatensatz mit sklearn
evaluate(numpy.argmax(pred, axis=1), numpy.argmax(test_data_labels, axis=1))


