import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model = load_model('rock_genre_classifier.h5')

test_data_path = '/homes/xdevore/ml-final-project/ml-final/data/test/rock_test_specs'
batch_size = 32

test_generator = custom_generator(test_data_path, target_size=(128, 128), batch_size=batch_size, subset="test")

other_genres_folders = ['house', 'jazz', 'rap']
other_genres_count = sum([len(os.listdir(os.path.join(test_data_path, genre))) for genre in other_genres_folders])
rock_count = len(os.listdir(os.path.join(test_data_path, 'rock')))
test_steps = (rock_count + other_genres_count) // (batch_size * 2)

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
