import tensorflow as tf
import numpy as np
import gzip
import pickle
from model import custom_generator
# Load the pre-trained models
model_0 = tf.keras.models.load_model('path/to/your/model_0_file.h5')
model_1 = tf.keras.models.load_model('path/to/your/model_1_file.h5')

# Create the custom generator
test_generator = custom_generator(test_dir, target_size, batch_size, 'validation')

# Create a custom model to extract activations at the conv2 layer
def get_intermediate_model(model, layer_name):
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model

layer_name = 'name_of_your_conv2_layer'
model_0_intermediate = get_intermediate_model(model_0, layer_name)
model_1_intermediate = get_intermediate_model(model_1, layer_name)

# Get activations for the test data using the generator
acts1 = []
acts2 = []
num_batches = len(test_generator)
for i in range(num_batches):
    batch_data, _ = next(test_generator)
    batch_acts1 = model_0_intermediate.predict(batch_data)
    batch_acts2 = model_1_intermediate.predict(batch_data)
    acts1.append(batch_acts1)
    acts2.append(batch_acts2)

# Concatenate activations from all batches
acts1 = np.concatenate(acts1, axis=0)
acts2 = np.concatenate(acts2, axis=0)

# Save the activations
with gzip.open("./model_activations/SVHN/model_0_lay03.p", "wb") as f:
    pickle.dump(acts1, f)

with gzip.open("./model_activations/SVHN/model_1_lay03.p", "wb") as f:
    pickle.dump(acts2, f)

print(acts1.shape, acts2.shape)
