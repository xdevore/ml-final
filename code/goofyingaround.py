# import numpy as np
# arr = np.load('activations_matrix_jazz.npy')
# arr1 = np.load('activations_matrix_rock.npy')
# print(arr1.shape,arr.shape)
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
#from custom_generator import *
import os
import random
from PIL import Image
import cv2


#-----------------------------------------------------------
# Assuming your pretrained CNN is saved as 'pretrained_cnn.h5'

def activation_maker(activation_model,test_data):
    data = test_data
    activations = []


    activation_batch = activation_model.predict(data)
    activations.append(activation_batch.tolist())
    return activations


def get_layer(model_file, layer_name):
    model = load_model(model_file)



    layer_output = None

    for layer_cnn in model.layers:
        if layer_cnn.name == layer_name:

            layer_output = layer_cnn.output
            break


    if layer_output is None:
        raise ValueError(f"Layer with name '{layer_name}' not found in the model.")


    # Create a new model that outputs the activations from the desired layer


    activation_model = Model(inputs=model.input, outputs=layer_output)
    return activation_model

def get_random_files(folder_path, num_files):
    all_files = os.listdir(folder_path)
    random_files = random.sample(all_files, num_files)
    return [os.path.join(folder_path, file) for file in random_files]

def process_file(file_path,size):
    image = cv2.imread(file_path)

    # Resize the image to the desired dimensions
    image = cv2.resize(image, (192, 64))

    # Normalize pixel values to the range [0, 1]
    image = image / 255.0

    # Add an extra dimension to make it (1, 64, 192, 3)
    image = np.expand_dims(image, axis=0)

    return image

pretrained_cnns = ['house_genre_classifier.h5','rap_genre_classifier.h5','rock_genre_classifier.h5','rock_genre_classifier2.h5']

pretrained_cnn1 = load_model('rock_genre_classifier2.h5')
pretrained_cnn2 = load_model('rock_genre_classifier.h5')

folder_paths = ['../data/test/house_test_specs', '../data/test/rap_test_specs', '../data/test/rock_test_specs', '../data/test/jazz_test_specs']
num_files = 2
directory = '/homes/xdevore/ml-final-project/ml-final/data/test/'
# Get 100 random files from each folder
selected_files = []
for folder_path in folder_paths:
    selected_files.extend(get_random_files(folder_path, num_files))

# Specify the layer name you want to get the activations from
layer_name = 'conv2d_1'
target_size = (64, 192)
for cnn in pretrained_cnns:
    for file in selected_files:


        p_file = process_file(file, target_size)
        activation_layer = get_layer(cnn, layer_name)
        activations = activation_maker(activation_layer,p_file)

    activations = np.array(activations)
    activations_matrix = np.vstack(activations)
    file_split = cnn.split("_")
    new_file_name = file_split[0]
    np.save('/homes/areichard/Desktop/ml-final/layer_activations/activations_matrix_' + new_file_name + '.npy', activations_matrix)
