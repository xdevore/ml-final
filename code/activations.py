import numpy as np
import tensorflow as tf
import gzip
import pickle

import os
from skimage import io

# Define the directory paths and labels for each genre
genre_folders = {
    'rock': 'path/to/rock/folder',
    'genre_2': 'path/to/genre_2/folder',
    'genre_3': 'path/to/genre_3/folder',
    'genre_4': 'path/to/genre_4/folder'
}

positive_examples = []
negative_examples = []

# Read and label spectrograms
for genre, folder in genre_folders.items():
    for filename in os.listdir(folder):
        if filename.endswith(".png"):  # Assuming spectrograms are saved as PNG images
            filepath = os.path.join(folder, filename)
            img = io.imread(filepath)

            # Add the loaded image and its label (1 for positive/rock and 0 for negative/others) to the corresponding list
            if genre == 'rock':
                positive_examples.append((img, 1))
            else:
                negative_examples.append((img, 0))

# Convert lists to NumPy arrays for easier processing
positive_examples = np.array(positive_examples, dtype=object)
negative_examples = np.array(negative_examples, dtype=object)

# Randomly select samples
num_positive = 500
num_negative = 500 // 3

positive_indices = np.random.choice(len(positive_examples), num_positive, replace=False)
negative_indices = np.random.choice(len(negative_examples), num_negative * 3, replace=False)

selected_positive = positive_examples[positive_indices]
selected_negative = negative_examples[negative_indices]

# Combine and shuffle the samples
final_dataset = np.concatenate((selected_positive, selected_negative))
np.random.shuffle(final_dataset)

# Separate the images and labels
X_test = np.array([example[0] for example in final_dataset])
y_test = np.array([example[1] for example in final_dataset])

# Normalize the images if needed
X_test = X_test.astype('float32') / 255





# now I need to make the actual actications -----------------------------------------------------------------------

# Load or create your trained CNNs
model_0 = tf.keras.models.load_model('path/to/your/model_0_file.h5')
model_1 = tf.keras.models.load_model('path/to/your/model_1_file.h5')

# Load your test data (X_test)
# Assuming you have already loaded or created your test dataset, X_test with shape (1000, 32, 32, 3)

# Create a custom model to extract activations at the conv2 layer
def get_intermediate_model(model, layer_name):
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model

layer_name = 'name_of_your_conv2_layer'
model_0_intermediate = get_intermediate_model(model_0, layer_name)
model_1_intermediate = get_intermediate_model(model_1, layer_name)

# Get activations for the test data
acts1 = model_0_intermediate.predict(X_test)
acts2 = model_1_intermediate.predict(X_test)

# Save the activations
with gzip.open("./model_activations/SVHN/model_0_lay03.p", "wb") as f:
    pickle.dump(acts1, f)

with gzip.open("./model_activations/SVHN/model_1_lay03.p", "wb") as f:
    pickle.dump(acts2, f)

print(acts1.shape, acts2.shape)
