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
    #print("the sahep", data.shape)



    activation_batch = activation_model.predict(data)

    return activation_batch


def get_layer(model_file, layer_name):
    model = load_model(model_file)



    layer_output = None

    right_layer = model.layers[2]


    layer_output = right_layer.output



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
models_path ='/homes/areichard/Desktop/ml-final/onevsone_models/'
pretraining = os.listdir(models_path)
pretrained_cnns = []

for i in pretraining:
    pretrained_cnns.append(models_path + i)
print(pretrained_cnns)

pretrained_cnns = ["/homes/areichard/Desktop/ml-final/onevsone_models/house_vs_jazz0_genre_classifier.h5", "/homes/areichard/Desktop/ml-final/onevsone_models/house_vs_jazz0_genre_classifier.h5"]



#pretrained_cnn1 = load_model('rock_genre_classifier2.h5')
#pretrained_cnn2 = load_model('rock_genre_classifier.h5')

folder_paths = ['../data/test/house_test_specs', '../data/test/rap_test_specs', '../data/test/rock_test_specs', '../data/test/jazz_test_specs']
num_files = 100
directory = '/homes/xdevore/ml-final-project/ml-final/data/test/'
# Get 100 random files from each folder
selected_files = []
for folder_path in folder_paths:
    selected_files.extend(get_random_files(folder_path, num_files))

# Specify the layer name you want to get the activations from
layer_name = 'conv2d_'
layer_num = -2
target_size = (64, 192)
for cnn in pretrained_cnns:
    layer_num = layer_num +3
    activations = []
    for file in selected_files:


        p_file = process_file(file, target_size)
        activation_layer = get_layer(cnn, layer_name + str(layer_num))
        activation_batch = activation_maker(activation_layer,p_file)
        activations.append(activation_batch.tolist())

    activations = np.array(activations)
    activations_matrix = np.vstack(activations)
    print(activations_matrix.shape)
#    print("MAKETHISSEEENNNN_____________________________________________________\n\n\n", activations.shape)
    splitting = cnn.split("/")
    file_split = splitting[-1].split("_")


    #np.save('/homes/areichard/Desktop/ml-final/onevsone_activations/onevsone_activations_' + file_split[0] + "_vs_" + file_split[2] + "_ready", activations_matrix)
    np.save('onevsone_activations_' + file_split[0] + "_vs_" + file_split[2] + "_ready", activations_matrix)
