import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('rock_genre_classifier5.h5')
print(model.summary())
# Define image dimensions and genre folders
IMG_HEIGHT = 64
IMG_WIDTH = 192
genre_folders = ['house_test_specs', 'jazz_test_specs', 'rap_test_specs', 'rock_test_specs']

# Define test data folder
test_data_folder = '/homes/xdevore/ml-final-project/ml-final/data/test'

# Function to preprocess the spectrograms
def preprocess_spectrogram(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Initialize variables to track correct predictions and total predictions
correct_predictions = 0
total_predictions = 0

# Test the model on each genre
for genre in genre_folders:
    genre_folder = os.path.join(test_data_folder, genre)
    for img_file in os.listdir(genre_folder):
        img_path = os.path.join(genre_folder, img_file)
        img_array = preprocess_spectrogram(img_path)

        prediction = model.predict(img_array)
        is_rock = prediction[0][0] > 0.5

        if genre == 'rock_test_specs' and is_rock:
            correct_predictions += 1
        elif genre != 'rock_test_specs' and not is_rock:
            correct_predictions += 1
        total_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy * 100:.2f}%")
