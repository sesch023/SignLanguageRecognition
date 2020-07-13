from tensorflow.keras.preprocessing import image_dataset_from_directory

# Bezeichnungen der Ordner der Klassen.
class_names = [
    "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "10", "11",
    "12", "13", "14", "15", "16", "17",
    "18", "19", "20", "21", "22", "23",
    "24"
]
# Standardmäßige Bildgröße.
image_size = (28, 28)


def get_train_and_val(batch_size, shuffle=True, validation_split=0.2, seed=0, image_size_param=image_size):
    """
    Gibt Trainings- und Validerungsdaten in einem Tensorflow Batched-Dataset für den Kaggle Sign Language MNIST
    Datensatz zurück (https://www.kaggle.com/datamunge/sign-language-mnist), welcher die Bilder nacheinander in den Speicher laden
    kann. Diese werden aus dem Ordner train entnommen.
    :param batch_size: Größe der Batches.
    :param shuffle: Werden die Trainingsdaten zufällig angeordnet?
    :param validation_split: Der Anteil der Validierungsdaten.
    :param seed: Der Seed für die Zufällige Anordnung.
    :param image_size_param: Die Größe auf die die Bilder skaliert werden.
    :return: Trainingsdaten als Batched-Dataset, Validierungsdaten als Batched-Dataset
    """
    return image_dataset_from_directory(
        directory="datasets/kaggle_1/DatasetPrepared/train",
        label_mode='categorical',
        color_mode='grayscale',
        image_size=image_size_param,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        interpolation='bilinear',
        follow_links=False
    ), image_dataset_from_directory(
        directory="datasets/kaggle_1/DatasetPrepared/train",
        label_mode='categorical',
        color_mode='grayscale',
        image_size=image_size_param,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
        interpolation='bilinear',
        follow_links=False
    )


def get_test(image_size_param=image_size):
    """
    Gibt Testdaten in einem Tensorflow Batched-Dataset für den Kaggle Sign Language MNIST
    Datensatz zurück (https://www.kaggle.com/datamunge/sign-language-mnist), welcher die Bilder nacheinander in den Speicher laden
    kann. Diese werden aus dem Ordner test entnommen.
    :param image_size_param: Die Größe auf die die Bilder skaliert werden.
    :return: Testdaten als Batched-Dataset
    """
    return image_dataset_from_directory(
        directory="datasets/kaggle_1/DatasetPrepared/test",
        label_mode='categorical',
        color_mode='grayscale',
        image_size=image_size_param,
        batch_size=32,
        shuffle=False,
        seed=None,
        validation_split=None,
        interpolation='bilinear',
        follow_links=False
    )
