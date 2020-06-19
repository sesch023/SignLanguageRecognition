from PIL import Image
import csv
import os
import shutil
import numpy

train_file = "DatasetOrig/sign_mnist_train.csv"
test_file = "DatasetOrig/sign_mnist_test.csv"

target_dir = "DatasetPrepared/"

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

train_target_folder = target_dir + "train/"
test_target_folder = target_dir + "test/"

os.makedirs(target_dir, exist_ok=True)
os.makedirs(train_target_folder, exist_ok=True)
os.makedirs(test_target_folder, exist_ok=True)


def to_image_at_dir(target_dir, row_data, image_name):
    label = row_data[0]
    list_data = list(map(int, row_data[1::]))
    image_data = numpy.reshape(numpy.array(list_data, dtype=int), (28, 28))
    label_folder = target_dir + str(label) + "/"

    os.makedirs(label_folder, exist_ok=True)
    image = Image.fromarray(image_data)
    image.convert("RGB").save(label_folder + image_name + ".png")

train_image_num = 0

with open(train_file, "r") as csv_file:
    train_data = csv.reader(csv_file, delimiter=",")
    head = True

    for row in train_data:
        if head:
            head = False
            continue
        else:
            to_image_at_dir(train_target_folder, row, str(train_image_num))
            train_image_num += 1

test_image_num = 0

with open(test_file, "r") as csv_file:
    test_data = csv.reader(csv_file, delimiter=",")

    head = True

    for row in test_data:
        if head:
            head = False
            continue
        else:
            to_image_at_dir(test_target_folder, row, str(test_image_num))
            test_image_num += 1
