from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from sign_language_image.datasets.kaggle_1.DatasetParser import get_train_and_val, get_test, class_names
from model_logging.LogRun import log_run

batch_size = 100
image_size = (28, 28)
train_ds, val_ds = get_train_and_val(batch_size, image_size_param=image_size)
test_ds = get_test(image_size_param=image_size)

num_of_classes = len(class_names)
input_shape = (image_size[0], image_size[1], 1)

model = Sequential()
model.add(Conv2D(8, (3, 3), activation='elu', input_shape=input_shape, strides=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='elu', strides=(1, 1)))
model.add(Conv2D(32, (3, 3), activation='elu', strides=(1, 1)))
model.add(Conv2D(64, (3, 3), activation='elu', strides=(1, 1)))
model.add(Conv2D(128, (3, 3), activation='elu', strides=(1, 1)))
model.add(Flatten())
model.add(Dense(250, activation='elu'))
model.add(Dropout(0.25))
model.add(Dense(num_of_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.00005),
              metrics=['accuracy'])

tensorboard_callback, checkpoint, run_path = log_run("C:/Users/sebas/PycharmProjects/SignLanguageReconition/logs/")
model.summary()

model.fit(train_ds,
          epochs=50,
          verbose=1,
          validation_data=val_ds,
          callbacks=[tensorboard_callback, checkpoint])
model = load_model(run_path + "model.mdl_wts.hdf5")
score = model.evaluate(test_ds, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
