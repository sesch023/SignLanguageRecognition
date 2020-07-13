from PIL import Image
import csv
import os
import shutil
import numpy

"""
Dieses Skript verarbeitet die Daten für den Kaggle Sign Language MNIST
Datensatz (https://www.kaggle.com/datamunge/sign-language-mnist), welche im Ordner
DatasetOrig vorliegen müssen, so, dass sie in passenden Ordnerstrukturen von Test und 
Trainingsdaten vorliegen. Ziel davon ist es, dass image_dataset_from_directory genutzt 
werden kann. Dazu müssen die Bilder einer Klasse in einzelnen passenden Ordner vorliegen.
"""

# CSV Dateien
train_file = "DatasetOrig/sign_mnist_train.csv"
test_file = "DatasetOrig/sign_mnist_test.csv"

# Ziel Directory
target_dir = "DatasetPrepared/"

# Ziel Directory entfernen
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# Sub Directories
train_target_folder = target_dir + "train/"
test_target_folder = target_dir + "test/"

# Ziel Directories erstellen
os.makedirs(target_dir, exist_ok=True)
os.makedirs(train_target_folder, exist_ok=True)
os.makedirs(test_target_folder, exist_ok=True)


def to_image_at_dir(target_dir, row_data, image_name):
    """
    Wandelt die Zeilen einer CSV Datei im Datensatz in ein Bild um.
    :param target_dir: Zielordner
    :param row_data: Daten der Zeile.
    :param image_name: Name des Bildes.
    :return: None.
    """
    # Label entnehmen.
    label = row_data[0]
    # Daten entnehmen in Integer umwandeln.
    list_data = list(map(int, row_data[1::]))

    # Image Data in Numpy 28x28 Array umwandeln.
    image_data = numpy.reshape(numpy.array(list_data, dtype=numpy.uint8), (28, 28))
    label_folder = target_dir + str(label) + "/"

    # Ordner erstellen, wenn nicht vorhanden.
    os.makedirs(label_folder, exist_ok=True)
    # Bild erstellen.
    image = Image.fromarray(image_data)
    # Bild als PNG speichern.
    image.convert("RGB").save(label_folder + image_name + ".png")

# Nummer des Bildes
train_image_num = 0

# CSV für Training lesen
with open(train_file, "r") as csv_file:
    train_data = csv.reader(csv_file, delimiter=",")
    head = True

    # Zeilen durchlaufen, Kopf ignorieren
    for row in train_data:
        if head:
            head = False
            continue
        else:
            # Zeile als Bild speichern.
            to_image_at_dir(train_target_folder, row, str(train_image_num))
            train_image_num += 1

test_image_num = 0

# CSV für Test lesen
with open(test_file, "r") as csv_file:
    test_data = csv.reader(csv_file, delimiter=",")

    head = True

    # Zeilen durchlaufen, Kopf ignorieren
    for row in test_data:
        if head:
            head = False
            continue
        else:
            # Zeile als Bild speichern
            to_image_at_dir(test_target_folder, row, str(test_image_num))
            test_image_num += 1
