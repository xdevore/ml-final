import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
import os
from custom_generator import *


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
data_type = "test"

# Create the custom generator
print("4")
generator = custom_generator(directory, target_size, batch_size, subset, data_type)

print("made ittt")

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
