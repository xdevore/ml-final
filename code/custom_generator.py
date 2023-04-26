import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def custom_generator(directory, target_size, batch_size, subset, data_type, genre_train_folder=None):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    other_genres_folders = []
    other_genres_generators = []
    specified_genre_generator = None
    if data_type == "train":
        other_genres_folders = ['house_specs', 'jazz_specs', 'rap_specs', 'rock_specs']
        other_genres_folders[:other_genres_folders.index(genre_train_folder)] + other_genres_folders[other_genres_folders.index(genre_train_folder)+1:]

        specified_genre_generator = datagen.flow_from_directory(
        directory,
        classes=[genre_train_folder],
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="binary",
        subset=subset
        )

    else:
       other_genres_folders = ['house_test_specs', 'jazz_test_specs', 'rap_test_specs', 'rock_test_specs']

    for genre in other_genres_folders:
        print(f"Processing genre: {genre}")
        print(f"Looking for folder: {os.path.join(directory, genre)}")
    for genre in other_genres_folders:
        genre_generator = datagen.flow_from_directory(
            directory,
            classes=[genre],
            target_size=target_size,
            color_mode="rgb",
            batch_size=batch_size // len(other_genres_folders),
            class_mode="binary",
            subset=subset
        )
        other_genres_generators.append(genre_generator)


    while True:

        if data_type == "train":
            specific_genre_data, _ = next(specified_genre_generator)
            other_genres_data = []

            for genre_generator in other_genres_generators:
                genre_data, _ = next(genre_generator)
                other_genres_data.append(genre_data)

            other_genres_data = np.vstack(other_genres_data)
            data = np.vstack((specific_genre_data, other_genres_data))
            labels = np.vstack((np.ones((len(specific_genre_data), 1)), np.zeros((len(other_genres_data), 1))))

        else:
            other_genres_data = []
            other_genres_labels = []

            for genre_generator in other_genres_generators:

                genre_data, genre_labels = next(genre_generator)

            other_genres_data.append(genre_data)
            other_genres_labels.append(genre_labels)

            data = np.concatenate(other_genres_data, axis=0)
            labels = np.concatenate(other_genres_labels, axis=0)

        if data.size == 0 or labels.size == 0:
            print("Empty batch encountered")
            continue

        yield data, labels

def custom_generator_onevsone(directory, target_size, batch_size, subset, specified_genre_train_folder, other_genre_train_folder):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    specified_genre_generator = datagen.flow_from_directory(
        directory,
        classes=[specified_genre_train_folder],
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="binary",
        subset=subset
    )

    other_genre_generator = datagen.flow_from_directory(
            directory,
            classes=[other_genre_train_folder],
            target_size=target_size,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="binary",
            subset=subset
    )

    while True:

        specific_genre_data, _ = next(specified_genre_generator)
        other_genre_data, _ = next(other_genre_generator)

        data = np.vstack((specific_genre_data, other_genre_data))
        labels = np.vstack((np.ones((len(specific_genre_data), 1)), np.zeros((len(other_genre_data), 1))))


        if data.size == 0 or labels.size == 0:
            print("Empty batch encountered")
            continue

        yield data, labels
