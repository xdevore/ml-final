

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
import os


#----------------------just chill her for a sec while I figrue out what to do with your

def custom_generator(directory, target_size, batch_size, subset):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    other_genres_generators = []
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
#-----------------------------------------------------------
# Assuming your pretrained CNN is saved as 'pretrained_cnn.h5'
print("1")
pretrained_cnn1 = load_model('rock_genre_classifier2.h5')
pretrained_cnn2 = load_model('rock_genre_classifier.h5')

# Specify the layer name you want to get the activations from
layer_name = 'conv2d_1'

# Find the layer with the specified name
print("2")
layer_output1 = None
layer_output2 = None
for layer_cnn1, layer_cnn2 in zip(pretrained_cnn1.layers,pretrained_cnn2.layers):
    if layer_cnn1.name == layer_name:
        layer_output1 = layer_cnn1.output
        layer_output2 = layer_cnn2.output
        break
print("output shape of single",layer_output1.shape,layer_output2.shape)
if layer_output1 is None:
    raise ValueError(f"Layer with name '{layer_name}' not found in the model.")

# Create a new model that outputs the activations from the desired layer
print("3")
activation_model1 = Model(inputs=pretrained_cnn1.input, outputs=layer_output1)
activation_model2 = Model(inputs=pretrained_cnn2.input, outputs=layer_output2)

# Set your parameters
directory = '/homes/xdevore/ml-final-project/ml-final/data/test/'
target_size = (64, 192)
batch_size = 8
subset = 'validation'

# Create the custom generator
print("4")
generator = custom_generator(directory, target_size, batch_size, subset)

# Run activations through the pretrained CNN on 1000 test examples
activations1 = []
activations2 = []
num_batches = 500 // batch_size
print("5")
for i in range(num_batches):

    data, labels = next(generator)
    print("the data hasss this shape", data.shape)
    activation_batch1 = activation_model1.predict(data)
    activation_batch2 = activation_model2.predict(data)
    activations1.append(activation_batch1.tolist())
    activations2.append(activation_batch2.tolist())
activations1 = np.array(activations1)
activations2 = np.array(activations2)

# Stack activations on top of each other as a matrix
activations_matrix1 = np.vstack(activations1)
activations_matrix2 = np.vstack(activations2)

# Save the activations matrix
np.save('activations_matrix_rock2.npy', activations_matrix1)
np.save('activations_matrix_rock.npy', activations_matrix2)
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
