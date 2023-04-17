import numpy as np
import tensorflow as tf
import gzip
import pickle

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
