import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from custom_generator import *
import matplotlib.pyplot as plt

data_path = "/homes/xdevore/ml-final-project/ml-final/data/train/"
batch_size = 32
#(288, 432)

genre_list = ["house", "jazz", "rap", "rock"]
genres_folders = ['house_specs', 'jazz_specs', 'rap_specs', 'rock_specs']

for i in range(len(genre_list)):

    specified_genre = genre_list[i]
    specified_genre_specs_folder = ""
    for item in genres_folders:
        if specified_genre in item:
            specified_genre_specs_folder = item

    for j in range(i + 1, len(genre_list)):

        other_genre = genre_list[j]
        other_genre_specs_folder = ""
        for item in genres_folders:
            if other_genre in item:
                other_genre_specs_folder = item

        for k in range(3):
            train_generator = custom_generator_onevsone(data_path, target_size=(64, 192), batch_size=batch_size, subset="training", specified_genre_train_folder=specified_genre_specs_folder, other_genre_train_folder=other_genre_specs_folder)
            validation_generator = custom_generator_onevsone(data_path, target_size=(64, 192), batch_size=batch_size, subset="validation", specified_genre_train_folder=specified_genre_specs_folder, other_genre_train_folder=other_genre_specs_folder)

            other_genres_count = len(os.listdir(os.path.join(data_path, other_genre_specs_folder)))
            specified_genre_count = len(os.listdir(os.path.join(data_path, specified_genre_specs_folder)))

            print(other_genres_count)
            print(specified_genre_count)

            train_steps_per_epoch = (specified_genre_count + other_genres_count) // (batch_size * 2)
            validation_steps = train_steps_per_epoch // 5

            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 192, 3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

            model_name = specified_genre + '_vs_' + other_genre + '_best_model_' + str(k) + '.h5'
            checkpoint = ModelCheckpoint(model_name, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

            history = model.fit(
                train_generator,
                steps_per_epoch=train_steps_per_epoch,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                epochs=11,
                callbacks=[checkpoint]
            )

            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')

            plt.title("Loss Plot")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()

            plt.show()
            plt.savefig()

            model_name = specified_genre + '_vs_' + other_genre + str(k) + '_genre_classifier.h5'

            model.save(model_name)
