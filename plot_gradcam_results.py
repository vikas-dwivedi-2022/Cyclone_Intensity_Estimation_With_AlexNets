import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import keras
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,BatchNormalization,Flatten,Reshape,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.constraints import max_norm
from tensorflow.keras import layers
import matplotlib.cm as cm
from IPython.display import Image, display
from PIL import Image
from scipy.ndimage import zoom
from scipy import ndimage



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



# List of model paths
model_paths = [
    '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_11/Model_11_Global.h5',
    '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_12/Model_12_Global.h5',
    '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_13/Model_13_Global.h5',
    '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_14/Model_14_Global.h5',
    '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_15/Model_15_Global.h5',
    '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_16/Model_16_Global.h5',
    '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_17/Model_17_Global.h5',
    '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_18/Model_18_Global.h5',
    '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_19/Model_19_Global.h5',
    '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_20/Model_20_Global.h5'
]
#--------------------------------------#
#           Sample Image               #
#--------------------------------------#
img_path = "/home/vikas/Documents/Final_Dance/TEST/SPEED_WISE/140/fit/fit_053.jpg"

# List to store heatmaps
heatmaps = []

# Iterate over each model
for model_path in model_paths:
    # Read and preprocess image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 255
    img_ref = img
    img = img.reshape(1, 366, 366, 1).astype('float32')

    # Load model
    SB = load_model(model_path)

    # Predict speed
    speed = SB.predict(img)
    print(f"Speed for {os.path.basename(model_path)} = {speed[0]}")

    # Generate Grad-CAM heatmap
    last_conv_layer_name = "conv2d_4"  # Replace with the actual last convolutional layer name
    heatmap = make_gradcam_heatmap(img, SB, last_conv_layer_name)
    heatmap = np.array(heatmap)

    # Zoom operation
    target_shape = (366, 366)
    scale_factors = (target_shape[0] / heatmap.shape[0], target_shape[1] / heatmap.shape[1])
    heatmap = zoom(heatmap, scale_factors, order=3)

    # Save the heatmap in the list
    heatmaps.append(heatmap)


# Create a 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(14, 14))

# Iterate over the first 9 heatmaps and plot them
for i in range(9):
    row, col = divmod(i, 3)
    ax = axes[row, col]

    # Plot the heatmap
    im = ax.imshow(heatmaps[i], cmap='jet', aspect='equal')  # You can change the colormap if needed
    ax.set_title(f'Heatmap {i + 1}',fontsize=25)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Intensity',fontsize=20)

# Adjust layout and display the subplots
plt.tight_layout()
plt.savefig('hmap_tiled.jpg', dpi=300, bbox_inches='tight')
plt.show()




# Median Heatmap
average_heatmap = np.median(heatmaps, axis=0)

#----------------#
#     Plots       #
#----------------#
plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_ref,cmap='gray')
plt.title('Original Image',fontsize=25)

plt.subplot(1, 3, 2)
#plt.imshow(binary_mask, cmap='gray')
#plt.title('Median Mask',fontsize=25)
plt.imshow(average_heatmap, cmap='jet')
plt.title('Median Heatmap',fontsize=25)

plt.subplot(1, 3, 3)
img = plt.imshow(img_ref)
heatmap = plt.imshow(average_heatmap, cmap='jet', alpha=0.4)
#cbar = plt.colorbar(heatmap,label='Activation')
plt.title('Original+Heatmap ',fontsize=25)

plt.savefig('EYES.jpg', dpi=300, bbox_inches='tight')
plt.show()




