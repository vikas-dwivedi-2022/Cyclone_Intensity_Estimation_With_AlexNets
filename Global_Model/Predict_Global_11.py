#----------------------------------------------------------------------------------------------%
#                                          IMPORTS                                             %
#----------------------------------------------------------------------------------------------%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

#-----------------------------------------------------------------------------------------------------%
# Given: Trained Global Model and Paths of Test Images. To Do: Predict Wind Speed using Trained Model % 
#-----------------------------------------------------------------------------------------------------%
df_output = pd.DataFrame(columns=["Image ID","True","Pred"])
model_path = '/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/MODEL_11/Model_11_Global.h5' # Trained global model
SB = load_model(model_path) # Load the trained global model

# Define the path to the CSV file
csv_file_path = "/home/vikas/Documents/Final_Dance/PYFEAT_EXPERIMENT/TEST_IMAGES_PATH.csv" # Path of test images
#csv_file_path = "/home/vikas/Documents/Final_Dance/PYFEAT_EXPERIMENT/TRAIN_IMAGES_PATH.csv"
#csv_file_path = "/home/vikas/Documents/Final_Dance/PYFEAT_EXPERIMENT/VAL_IMAGES_PATH.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)


for i, (index, row) in enumerate(df.iterrows()):  
    img_path = row["ImagePaths"]

    # Extract the value after SPEED_WISE
    speed_true = img_path.split('/SPEED_WISE/')[1].split('/')[0]
    speed_true = float(speed_true)  # Convert to float
    image_id = os.path.basename(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)/255 # Normalization 
    img = img.reshape(1, 366, 366, 1).astype('float32') 
    
    
    speed = SB.predict(img)[0][0] # Prediction by the trained model 

    # Append the values to the DataFrame
    df_output = df_output.append({"Image ID": image_id, "True": speed_true, "Pred": speed}, ignore_index=True)
    

# Save the results    
pred_path = "Predictions_Test_11.csv"
#pred_path = "Predictions_TRAIN_11.csv"
#pred_path = "Predictions_VAL_11.csv"
df_output.to_csv(pred_path, index=False)  # Set index=False to exclude the index column

print(f"Predictions saved as CSV to '{pred_path}'")    
