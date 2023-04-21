

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
#----------------------just chill her for a sec while I figrue out what to do with your

def custom_generator(directory, target_size, batch_size, subset):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    rock_generator = datagen.flow_from_directory(
        directory,
        classes=['rock_test_specs'],
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="binary",
        subset=subset
    )

    other_genres_generators = []
    other_genres_folders = ['house_test_specs', 'jazz_test_specs', 'rap_test_specs']

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

#-----------------------------------------------------------
# Assuming your pretrained CNN is saved as 'pretrained_cnn.h5'
print("1")
pretrained_cnn = load_model('rock_genre_classifier.h5')

# Specify the layer name you want to get the activations from
layer_name = 'conv2d_1'

# Find the layer with the specified name
print("2")
layer_output = None
for layer in pretrained_cnn.layers:
    if layer.name == layer_name:
        layer_output = layer.output
        break
print("output shape of single",layer_output.shape)
if layer_output is None:
    raise ValueError(f"Layer with name '{layer_name}' not found in the model.")

# Create a new model that outputs the activations from the desired layer
print("3")
activation_model = Model(inputs=pretrained_cnn.input, outputs=layer_output)

# Set your parameters
directory = '/homes/xdevore/ml-final-project/ml-final/data/test/'
target_size = (64, 192)
batch_size = 1
subset = 'validation'

# Create the custom generator
print("4")
generator = custom_generator(directory, target_size, batch_size, subset)

# Run activations through the pretrained CNN on 1000 test examples
activations = []
num_batches = 500 // batch_size
print("5")
for i in range(num_batches):
    data, labels = next(generator)
    activation_batch = activation_model.predict(data)
    activations.extend(activation_batch)

# Stack activations on top of each other as a matrix
activations_matrix = np.vstack(activations)

# Save the activations matrix
np.save('activations_matrix_rock4.npy', activations_matrix)
# chunk_size = 50
# num_chunks = num_batches // chunk_size
#
# for chunk in range(num_chunks):
#     activations = []
#
#     for i in range(chunk_size):
#         data, labels = next(generator)
#         activation_batch = activation_model.predict(data)
#         activations.extend(activation_batch)
#     print("make it here")
#     # Stack activations on top of each other as a matrix
#     activations_matrix = np.vstack(activations)
#
#     # Save the activations matrix chunk
#     np.save(f'activations_matrix_chunk_{chunk}.npy', activations_matrix)
