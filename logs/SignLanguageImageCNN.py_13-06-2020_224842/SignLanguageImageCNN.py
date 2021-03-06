from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from sign_language_image.datasets.kaggle_2.DatasetParser import get_train_and_val, get_test, class_names, image_size
from model_logging.LogRun import log_run

batch_size = 100
train_ds, val_ds = get_train_and_val(batch_size)
test_ds = get_test()

num_of_classes = len(class_names)
input_shape = (image_size[0], image_size[1], 1)

model = Sequential()
model.add(Conv2D(8, (7, 7), activation='elu', input_shape=input_shape, strides=(2, 2)))
model.add(Conv2D(16, (7, 7), activation='elu', strides=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='elu', strides=(2, 2)))
model.add(Flatten())
model.add(Dense(250, activation='elu'))
model.add(Dropout(0.25))
model.add(Dense(num_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.00005),
              metrics=['accuracy'])

model.summary()

tensorboard_callback, checkpoint = log_run("C:/Users/sebas/PycharmProjects/SignLanguageReconition/logs/")
model.fit(train_ds,
          epochs=40,
          verbose=1,
          validation_data=val_ds,
          callbacks=[tensorboard_callback, checkpoint])
score = model.evaluate(test_ds, verbose=1, callbacks=[tensorboard_callback])
print('Test loss:', score[0])
print('Test accuracy:', score[1])
