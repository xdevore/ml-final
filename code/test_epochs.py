import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from custom_generator import *

data_path = "/homes/xdevore/ml-final-project/ml-final/data/train/"
batch_size = 32
#(288, 432)


specified_genre = "rock"
genres_folders = ['house_specs', 'jazz_specs', 'rap_specs', 'rock_specs']
specified_genre_specs_folder = ""
for item in genres_folders:
    if specified_genre in item:
        specified_genre_specs_folder = item


train_generator = custom_generator(data_path, target_size=(64, 192), batch_size=batch_size, subset="training", data_type="train", genre_train_folder=specified_genre_specs_folder)
validation_generator = custom_generator(data_path, target_size=(64, 192), batch_size=batch_size, subset="validation", data_type="train", genre_train_folder=specified_genre_specs_folder)

other_genres_folders = genres_folders[:genres_folders.index(specified_genre_specs_folder)] + genres_folders[genres_folders.index(specified_genre_specs_folder)+1:]

other_genres_count = sum([len(os.listdir(os.path.join(data_path, genre))) for genre in other_genres_folders])
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

checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=15,
    callbacks=[checkpoint]
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

model_name = specified_genre + '_genre_classifier.h5'

model.save(model_name)
