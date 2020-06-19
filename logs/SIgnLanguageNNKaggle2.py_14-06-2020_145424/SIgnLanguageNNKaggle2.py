from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from sign_language_image.datasets.kaggle_2.DatasetParser import get_train_and_val, get_test, class_names
from model_logging.LogRun import log_run

image_size = (50, 50)

batch_size = 100
train_ds, val_ds = get_train_and_val(batch_size, image_size_param=image_size)
test_ds = get_test(image_size_param=image_size)

num_of_classes = len(class_names)
input_shape = (image_size[0], image_size[1], 1)

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(tf.keras.layers.Dense(2048, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(tf.keras.layers.Dense(1024, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(tf.keras.layers.Dense(512, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(tf.keras.layers.Dense(128, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(tf.keras.layers.Dense(64, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(num_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.00005),
              metrics=['accuracy'])

tensorboard_callback, checkpoint, run_path = log_run("C:/Users/sebas/PycharmProjects/SignLanguageReconition/logs/")
model.summary()

model.fit(train_ds,
          epochs=40,
          verbose=1,
          validation_data=val_ds,
          callbacks=[tensorboard_callback, checkpoint])
model = load_model(run_path + "model.mdl_wts.hdf5")
score = model.evaluate(test_ds, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


