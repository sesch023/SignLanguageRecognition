from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Lambda
from sign_language_image.datasets.kaggle_2.DatasetParser import get_train_and_val, get_test, class_names
from model_logging.LogRun import log_run
from Util import evaluate
import numpy

# Add before any TF calls
# Initialize the keras global outside of any tf.functions
temp = tf.random.uniform([4, 32, 32, 3])  # Or tf.zeros
tf.keras.applications.vgg19.preprocess_input(temp)

batch_size = 100
image_size = (224, 224)
train_ds, val_ds = get_train_and_val(batch_size, image_size_param=image_size, color_mode="rgb")
test_ds = get_test(image_size_param=image_size, color_mode="rgb")

num_of_classes = len(class_names)

input_shape = (image_size[0], image_size[1], 3)

basemodel = tf.keras.applications.VGG19(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=1000
)
basemodel.trainable = False
model = Sequential()
model.add(Lambda(tf.keras.applications.vgg19.preprocess_input, name='preprocessing', input_shape=input_shape))
model.add(basemodel)
model.add(Flatten())
model.add(Dense(4096, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(1024, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(num_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.000005),
              metrics=['accuracy'])

tensorboard_callback, checkpoint, run_path = log_run("/home/sebastian/PycharmProject/SignLanguageRecognition/logs/",
                                                     True)
model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(train_ds,
          epochs=50,
          verbose=1,
          validation_data=val_ds,
          callbacks=[tensorboard_callback, checkpoint, es])
model = load_model(run_path + "model.mdl_wts.hdf5")

test_data_labels = []
for data, label in test_ds.take(-1):
    test_data_labels.extend(label)

pred = model.predict(test_ds, verbose=1)
evaluate(numpy.argmax(pred, axis=1), numpy.argmax(test_data_labels, axis=1))


