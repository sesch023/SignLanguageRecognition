import shutil
import os
import glob
import random

orig_directory = "DatasetOrig"
target_directory = "DatasetPrepared"

train_directory_name = "train"
test_directory_name = "test"

rand_num_to_move = 300

if os.path.exists(target_directory):
    shutil.rmtree(target_directory)

shutil.copytree(orig_directory, target_directory)

for target_dir_name in glob.glob(target_directory + "/" + test_directory_name + "/*"):
    new_path = target_dir_name.replace("_", "/")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    shutil.move(target_dir_name, new_path)

for target_dir_name in glob.glob(target_directory + "/" + train_directory_name + "/*"):
    target_dir_classes = glob.glob(target_dir_name + "/*")
    random_elements = random.sample(target_dir_classes, rand_num_to_move)
    for move_element in random_elements:
        new_dir = move_element.replace(train_directory_name, test_directory_name)
        os.makedirs(os.path.dirname(new_dir), exist_ok=True)
        shutil.move(move_element, new_dir)




