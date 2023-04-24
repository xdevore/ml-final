import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def custom_generator(directory, target_size, batch_size, subset):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    rock_generator = datagen.flow_from_directory(
        directory,
        classes=['house_specs'],
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="binary",
        subset=subset
    )

    other_genres_generators = []
    other_genres_folders = ['rock_specs', 'jazz_specs', 'rap_specs']

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
        rock_data, _ = next(rock_generator)
        other_genres_data = []

        for genre_generator in other_genres_generators:
            genre_data, _ = next(genre_generator)
            other_genres_data.append(genre_data)

        other_genres_data = np.vstack(other_genres_data)
        data = np.vstack((rock_data, other_genres_data))
        labels = np.vstack((np.ones((len(rock_data), 1)), np.zeros((len(other_genres_data), 1))))
        yield data, labels

data_path = "/homes/xdevore/ml-final-project/ml-final/data/train/"
batch_size = 32
#(288, 432)
train_generator = custom_generator(data_path, target_size=(64, 192), batch_size=batch_size, subset="training")
validation_generator = custom_generator(data_path, target_size=(64, 192), batch_size=batch_size, subset="validation")

other_genres_folders = ['rock_specs', 'jazz_specs','rap_specs']
other_genres_count = sum([len(os.listdir(os.path.join(data_path, genre))) for genre in other_genres_folders])
rock_count = len(os.listdir(os.path.join(data_path, 'house_specs')))

train_steps_per_epoch = (rock_count + other_genres_count) // (batch_size * 2)
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

checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=5,
    callbacks=[checkpoint]
)



model.save('house_genre_classifier.h5')
