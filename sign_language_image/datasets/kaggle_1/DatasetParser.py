from tensorflow.keras.preprocessing import image_dataset_from_directory

class_names = [
    "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "10", "11",
    "12", "13", "14", "15", "16", "17",
    "18", "19", "20", "21", "22", "23",
    "24"
]
image_size = (28, 28)


def get_train_and_val(batch_size, shuffle=True, validation_split=0.2, seed=0, image_size_param=image_size):
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