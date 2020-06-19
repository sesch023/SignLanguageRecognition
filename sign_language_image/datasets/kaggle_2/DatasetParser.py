from tensorflow.keras.preprocessing import image_dataset_from_directory

class_names = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X",
    "Y", "Z", "space", "del", "nothing"
]
image_size = (200, 200)


def get_train_and_val(batch_size, shuffle=True, validation_split=0.2, seed=0, image_size_param=image_size):
    return image_dataset_from_directory(
        directory="datasets/kaggle_2/DatasetPrepared/train",
        label_mode='categorical',
        color_mode='grayscale',
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
        color_mode='grayscale',
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


def get_test(image_size_param=image_size):
    return image_dataset_from_directory(
        directory="datasets/kaggle_2/DatasetPrepared/test",
        label_mode='categorical',
        color_mode='grayscale',
        class_names=class_names,
        image_size=image_size_param,
        batch_size=32,
        shuffle=False,
        seed=None,
        validation_split=None,
        interpolation='bilinear',
        follow_links=False
    )
