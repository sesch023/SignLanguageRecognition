from tensorflow.keras.preprocessing import image_dataset_from_directory

# Namen der Klassenordner
class_names = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X",
    "Y", "Z", "space", "del", "nothing"
]
# Größe der Bilder
image_size = (200, 200)


def get_train_and_val(batch_size, shuffle=True, validation_split=0.2, seed=0, image_size_param=image_size,
                      color_mode="grayscale"):
    """
    Gibt Trainings- und Validerungsdaten in einem Tensorflow Batched-Dataset für den Kaggle ASL-Alphabet Datensatz
    zurück (https://www.kaggle.com/grassknoted/asl-alphabet), welcher die Bilder nacheinander in den
    Speicher laden kann. Diese werden aus dem Ordner train entnommen.
    :param batch_size: Größe der Batches.
    :param shuffle: Werden die Trainingsdaten zufällig angeordnet?
    :param validation_split: Der Anteil der Validierungsdaten.
    :param seed: Der Seed für die Zufällige Anordnung.
    :param image_size_param: Die Größe auf die die Bilder skaliert werden.
    :param color_mode: Farbmodus der Bilder (grayscale oder rgb).
    :return: Trainingsdaten als Batched-Dataset, Validierungsdaten als Batched-Dataset
    """
    return image_dataset_from_directory(
        directory="datasets/kaggle_2/DatasetPrepared/train",
        label_mode='categorical',
        color_mode=color_mode,
        class_names=class_names,
        image_size=image_size_param,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        interpolation='bilinear',
        follow_links=False
    ), image_dataset_from_directory(
        directory="datasets/kaggle_2/DatasetPrepared/train",
        label_mode='categorical',
        color_mode=color_mode,
        class_names=class_names,
        image_size=image_size_param,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
        interpolation='bilinear',
        follow_links=False
    )


def get_test(image_size_param=image_size, color_mode="grayscale"):
    """
    Gibt Testdaten in einem Tensorflow Batched-Dataset für den Kaggle ASL-Alphabet Datensatz
    zurück (https://www.kaggle.com/grassknoted/asl-alphabet), welcher die Bilder nacheinander in den
    Speicher laden kann. Diese werden aus dem Ordner test entnommen.
    :param image_size_param: Die Größe auf die die Bilder skaliert werden.
    :param color_mode: Farbmodus der Bilder (grayscale oder rgb).
    :return: Testdaten als Batched-Dataset
    """
    return image_dataset_from_directory(
        directory="datasets/kaggle_2/DatasetPrepared/test",
        label_mode='categorical',
        color_mode=color_mode,
        class_names=class_names,
        image_size=image_size_param,
        batch_size=32,
        shuffle=False,
        seed=None,
        validation_split=None,
        interpolation='bilinear',
        follow_links=False
    )
