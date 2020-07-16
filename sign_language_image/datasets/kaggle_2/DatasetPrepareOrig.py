import shutil
import os
import glob
import random

"""
Dieses Skript verarbeitet die Daten für den Kaggle ASL-Alphabet
Datensatz (https://www.kaggle.com/grassknoted/asl-alphabet), welche im Ordner
DatasetOrig vorliegen müssen, so, dass sie in passenden Ordnerstrukturen von Test und 
Trainingsdaten vorliegen. Ziel davon ist es, dass image_dataset_from_directory genutzt 
werden kann. Dazu müssen die Bilder einer Klasse in einzelnen passenden Ordner vorliegen.

Um dieses Skript anzuwenden, muss der Datensatz lediglich entpackt werden, der oberste Ordner
in DatasetOrig umgenannt und die beiden Ordner für Trainings- und Testdaten in train, sowie
test umbenannt werden.
"""


# Original Ordner
orig_directory = "DatasetOrig"
target_directory = "DatasetPrepared"

# Name der Ordner für Training und Test
train_directory_name = "train"
test_directory_name = "test"

# Anzahl der Bilder, welche pro Klasse für Testdaten genutzt
# und zufällig umkopiert werden sollen.
rand_num_to_move = 300

# Entferne Target-Directory, wenn es schon vorhanden ist.
if os.path.exists(target_directory):
    shutil.rmtree(target_directory)

# Kopiere Orig-Dir in Target-Dir
shutil.copytree(orig_directory, target_directory)

# Erstelle für jeden File im Testordner ein eigenen Ordner und kopiere ihn in diesen.
for target_dir_name in glob.glob(target_directory + "/" + test_directory_name + "/*"):
    new_path = target_dir_name.replace("_", "/")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    shutil.move(target_dir_name, new_path)

# Bewege die in rand_num_to_move definierte Anzahl zufälliger Elemente pro
# Klasse in den Testordner.
for target_dir_name in glob.glob(target_directory + "/" + train_directory_name + "/*"):
    target_dir_classes = glob.glob(target_dir_name + "/*")
    # Wähle für diese Klasse rand_num_to_move zufällige Elemente aus.
    random_elements = random.sample(target_dir_classes, rand_num_to_move)
    # Kopiere die Elemente um.
    for move_element in random_elements:
        new_dir = move_element.replace(train_directory_name, test_directory_name)
        os.makedirs(os.path.dirname(new_dir), exist_ok=True)
        shutil.move(move_element, new_dir)




