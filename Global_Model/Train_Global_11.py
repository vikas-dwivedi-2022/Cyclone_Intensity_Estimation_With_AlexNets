#---------------------------------------------------------------------------------------------------------------------#
#                                                  IMPORTS                                                            #
#---------------------------------------------------------------------------------------------------------------------#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,BatchNormalization,Flatten,Reshape,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras
from keras.regularizers import l2
from keras.constraints import max_norm
from tensorflow.keras import layers

#---------------------------------------------------------------------------------------------------------------------#
#                                             ML flow init                                                            #
#---------------------------------------------------------------------------------------------------------------------#
# Why? To keep a log of training history
import mlflow
import mlflow.tensorflow
import shutil


tf.keras.backend.clear_session()


#---------------------------------------------------------------------------------------------------------------------#
#                                        BATCH-RELATED MODULES                                                        #
#---------------------------------------------------------------------------------------------------------------------#

# What is this? To create a csv file containing paths of images selected fot the training batch. 
# It can be verifed that the batch is unbiased towards any specific wind speed or cyclone event.

def train_batch_info():
    # Updated base_directory
    base_directory = "/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/ENS_11/TRAIN/"  # Here ENS_11 is just first instance. Similarly, we have ENS_11 to ENS_20.

    # Create a list to store DataFrames
    dfs = []

    # Iterate through all subfolders
    for subfolder in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, subfolder)

        if os.path.isdir(subfolder_path):
            # List sub-subfolders in the current subfolder
            sub_subfolders = [sub_subfolder for sub_subfolder in os.listdir(subfolder_path)
                             if os.path.isdir(os.path.join(subfolder_path, sub_subfolder))]

            if sub_subfolders:
                # Randomly select a sub-subfolder
                selected_sub_subfolder = random.choice(sub_subfolders)
                selected_sub_subfolder_path = os.path.join(subfolder_path, selected_sub_subfolder)

                # List csv files in the selected sub-subfolder
                csv_files = [csv_file for csv_file in os.listdir(selected_sub_subfolder_path)
                             if csv_file.lower().endswith(".csv")]

                if csv_files:
                    # Randomly pick a csv file from the selected sub-subfolder
                    chosen_csv = random.choice(csv_files)
                    chosen_csv_path = os.path.join(selected_sub_subfolder_path, chosen_csv)

                    # Read the csv file
                    df_csv = pd.read_csv(chosen_csv_path)

                    if len(df_csv) > 1:
                        # Randomly select a row (excluding the first row with column names)
                        chosen_row = df_csv.iloc[random.randint(1, len(df_csv) - 1)]
                    else:
                        # If there is only one row, choose the only available row (index 0)
                        chosen_row = df_csv.iloc[0]

                    # Extract the image path from the chosen row
                    chosen_image_path = chosen_row["Image Paths"]

                    # Create a DataFrame with the chosen image path, subfolder name, and sub_subfolder name
                    df = pd.DataFrame({
                        "Image Path": [chosen_image_path],
                        "Wind Speed": [subfolder],
                        "Event": [selected_sub_subfolder]
                    })
                    dfs.append(df)

    # Concatenate DataFrames in the list
    df = pd.concat(dfs, ignore_index=True)

    # Save the DataFrame to a CSV file
    csv_file = "Train_Batch_Info_03.csv"
    df.to_csv(csv_file, index=False)
#---------------------------------------------------------------------------------------------------------------------#
# What is this? To create a csv file containing paths of images selected for the validation batch. 
# Please note that we are not using this data for training the model.

def val_batch_info():
    # Updated base_directory
    base_directory = "/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/ENS_11/VAL/"

    # Create a list to store DataFrames
    dfs = []

    # Iterate through all subfolders
    for subfolder in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, subfolder)

        if os.path.isdir(subfolder_path):
            # List sub-subfolders in the current subfolder
            sub_subfolders = [sub_subfolder for sub_subfolder in os.listdir(subfolder_path)
                             if os.path.isdir(os.path.join(subfolder_path, sub_subfolder))]

            if sub_subfolders:
                # Randomly select a sub-subfolder
                selected_sub_subfolder = random.choice(sub_subfolders)
                selected_sub_subfolder_path = os.path.join(subfolder_path, selected_sub_subfolder)

                # List csv files in the selected sub-subfolder
                csv_files = [csv_file for csv_file in os.listdir(selected_sub_subfolder_path)
                             if csv_file.lower().endswith(".csv")]

                if csv_files:
                    # Randomly pick a csv file from the selected sub-subfolder
                    chosen_csv = random.choice(csv_files)
                    chosen_csv_path = os.path.join(selected_sub_subfolder_path, chosen_csv)

                    # Read the csv file
                    df_csv = pd.read_csv(chosen_csv_path)

                    if len(df_csv) > 1:
                        # Randomly select a row (excluding the first row with column names)
                        chosen_row = df_csv.iloc[random.randint(1, len(df_csv) - 1)]
                    else:
                        # If there is only one row, choose the only available row (index 0)
                        chosen_row = df_csv.iloc[0]

                    # Extract the image path from the chosen row
                    chosen_image_path = chosen_row["Image Paths"]

                    # Create a DataFrame with the chosen image path, subfolder name, and sub_subfolder name
                    df = pd.DataFrame({
                        "Image Path": [chosen_image_path],
                        "Wind Speed": [subfolder],
                        "Event": [selected_sub_subfolder]
                    })
                    dfs.append(df)

    # Concatenate DataFrames in the list
    df = pd.concat(dfs, ignore_index=True)

    # Save the DataFrame to a CSV file
    csv_file = "Val_Batch_Info_03.csv"
    df.to_csv(csv_file, index=False)
#---------------------------------------------------------------------------------------------------------------------#      
# What is this? To create a csv file containing paths of images selected for the test batch. 
# Please note that we are not using this data for training the model. 

def test_batch_info():
    # Updated base_directory
    base_directory = "/home/vikas/Documents/Final_Dance/ENSEMBLE_DATA/ENS_11/TEST/"

    # Create a list to store DataFrames
    dfs = []

    # Iterate through all subfolders
    for subfolder in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, subfolder)

        if os.path.isdir(subfolder_path):
            # List sub-subfolders in the current subfolder
            sub_subfolders = [sub_subfolder for sub_subfolder in os.listdir(subfolder_path)
                             if os.path.isdir(os.path.join(subfolder_path, sub_subfolder))]
            
            if sub_subfolders:
                # Randomly select a sub-subfolder
                selected_sub_subfolder = random.choice(sub_subfolders)
                selected_sub_subfolder_path = os.path.join(subfolder_path, selected_sub_subfolder)
                
                # List csv files in the selected sub-subfolder
                csv_files = [csv_file for csv_file in os.listdir(selected_sub_subfolder_path)
                             if csv_file.lower().endswith(".csv")]

                if csv_files:
                    
                    # Randomly pick a csv file from the selected sub-subfolder
                    chosen_csv = random.choice(csv_files)
                    chosen_csv_path = os.path.join(selected_sub_subfolder_path, chosen_csv)
                    
                    # Read the csv file
                    df_csv = pd.read_csv(chosen_csv_path)

                    if len(df_csv) > 1:
                        # Randomly select a row (excluding the first row with column names)
                        chosen_row = df_csv.iloc[random.randint(1, len(df_csv) - 1)]
                    else:
                        # If there is only one row, choose the only available row (index 0)
                        chosen_row = df_csv.iloc[0]

                    # Extract the image path from the chosen row
                    chosen_image_path = chosen_row["Image Paths"]

                    # Create a DataFrame with the chosen image path, subfolder name, and sub_subfolder name
                    df = pd.DataFrame({
                        "Image Path": [chosen_image_path],
                        "Wind Speed": [subfolder],
                        "Event": [selected_sub_subfolder]
                    })
                    dfs.append(df)

    # Concatenate DataFrames in the list
    df = pd.concat(dfs, ignore_index=True)

    # Save the DataFrame to a CSV file
    csv_file = "Test_Batch_Info_03.csv"
    df.to_csv(csv_file, index=False)
#---------------------------------------------------------------------------------------------------------------------# 
# What is this? This module was used to get access of masked cyclone images. In this algorithm, we are not working with masked images.
def update_batch_info(input_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Create the "Mask Path" column by appending "_mask" to the "Image Path" file names
    df["Mask Path"] = df["Image Path"].apply(lambda x: x.replace(".jpg", "_mask.jpg"))

    # Overwrite the existing CSV file with the updated DataFrame
    df.to_csv(input_file, index=False)
    
#---------------------------------------------------------------------------------------------------------------------#
# What is this? This module loads the training, validation and test batch using the image paths.
def load_batch(csv_file):
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)

    # Initialize lists to store loaded images as floats and their corresponding labels
    X_batch = []
    X_mask_batch=[]
    y_batch = []

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        image_path = row["Image Path"]
        wind_speed = row["Wind Speed"]
        mask_path = row["Mask Path"]

        # Load the image using OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Convert the image format from uint8 to float
        img_as_float = img.astype(float)
        mask_as_float = mask.astype(float)
        

        # Append 
        X_batch.append(img_as_float)
        X_mask_batch.append(mask_as_float)
        y_batch.append(wind_speed)
        
    return X_batch,X_mask_batch,y_batch

#---------------------------------------------------------------------------------------------------------------------#
#                                             ALEXNET                                                                 #
#---------------------------------------------------------------------------------------------------------------------#
# What is this? This is the slightly modified AlexNet model with regularization and dropout to avoid overfitting. 

model = models.Sequential()

model.add(layers.Conv2D(16, (3,3), kernel_initializer='he_uniform', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), activation='relu', padding='same', input_shape=(366, 366, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(BatchNormalization())
model.add(layers.Conv2D(32, (3,3), kernel_initializer='he_uniform', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), activation='relu', padding='same' ))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(BatchNormalization())
model.add(layers.Conv2D(64, (3,3), kernel_initializer='he_uniform', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), activation='relu', padding='same' ))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(BatchNormalization())
model.add(layers.Conv2D(128, (3,3), kernel_initializer='he_uniform', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), activation='relu', padding='same' ))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(BatchNormalization())
model.add(layers.Conv2D(256, (3,3), kernel_initializer='he_uniform', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), activation='relu', padding='same' ))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(layers.Flatten())
model.add(Dropout(0.3))
model.add(layers.Dense(128, kernel_initializer='he_uniform', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), activation='relu'))
model.add(layers.Dense(32, kernel_initializer='he_uniform', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), activation='relu'))
model.add(layers.Dense(1, kernel_initializer='he_uniform', activation='linear'))
model = models.Model(inputs=model.input, outputs=model.output) 

model.summary()

# Compile the model
# What is this? Optimizer is Adam and loss function is MSLE as described in the paper.
iniLR = 0.001
opt = Adam(learning_rate=iniLR, amsgrad=True) #Default lr is 0.001
#model.compile(optimizer=opt,loss='mae')
model.compile(optimizer=opt,loss='msle', metrics=['mae'])

#---------------------------------------------------------------------------------------------------------------------#
#                                              MLFLOW EXP                                                             #
#---------------------------------------------------------------------------------------------------------------------#

# Set up MLflow experiment
mlflow.set_experiment("Image-Regression-Experiment")

#---------------------------------------------------------------------------------------------------------------------#
#                                                TRAINING                                                             #
#---------------------------------------------------------------------------------------------------------------------#

# Start MLflow run
with mlflow.start_run():
    
    count=0
    epochs = 500
    validation_interval = 100
    min_val_loss = float('inf')


    for sub_epoch in range(epochs):
    
        for iteration in range(validation_interval):
            #---------------------------------------------------------------------------------------------------->
            #-------------------------#
            #    Training Data        #
            #-------------------------#
            train_batch_info()
            update_batch_info("Train_Batch_Info_03.csv")
            X_batch_train,X_mask_batch_train,y_batch_train = load_batch("Train_Batch_Info_03.csv")        

            #-------------------------#
            #    Augmentation         #
            #-------------------------#
            # What is this? Data augmentation using rotation.
            rows, cols = X_batch_train[0].shape
            cx, cy = cols / 2, rows / 2
            for i in range(len(X_batch_train)):
                if np.random.rand() > 0.5:    
                    angle = np.random.uniform(-180, 180)
                    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1) 
                    X_batch_train[i] = cv2.warpAffine(X_batch_train[i], rotation_matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT)
                    X_mask_batch_train[i] = cv2.warpAffine(X_mask_batch_train[i], rotation_matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        
            X_batch_train=np.array(X_batch_train)/255
            X_mask_batch_train=np.array(X_mask_batch_train)/255
            y_batch_train=np.array(y_batch_train)
        
            #-------------------------#
            #    Training             #
            #-------------------------#     
            train_loss=model.train_on_batch(X_batch_train, y_batch_train)
        
            print("Iteration {}/{} (Epoch {}/{}) - Training Loss: {:.4f}".format(iteration + 1, validation_interval, sub_epoch + 1, epochs, train_loss[1]))
        
            
            mlflow.log_metric("Training Loss", train_loss[1])
        
        
            #---------------------------------------------------------------------------------------------------->
            #----------------------#
            #    Validation Data   #
            #----------------------#
            if (iteration + 1) % validation_interval == 0:
                val_batch_info()
                update_batch_info("Val_Batch_Info_03.csv")
                X_batch_val,X_mask_batch_val,y_batch_val = load_batch("Val_Batch_Info_03.csv")
            
                X_batch_val = np.array(X_batch_val)/255     
                X_mask_batch_val = np.array(X_mask_batch_val)/255
                y_batch_val=np.array(y_batch_val)
                    
            #----------------------#
            #    Validation        #
            #----------------------#     
                val_loss=model.test_on_batch(X_batch_val, y_batch_val) 
                print("<------------------------------------------------------------------------------------->")
                print("Iteration {}/{} (Epoch {}/{}) - Validation Loss: {:.4f}".format(iteration + 1, validation_interval, sub_epoch + 1, epochs, val_loss[1]))
                
                mlflow.log_metric("Validation Loss", val_loss[1])
                
            
                if val_loss[1] < min_val_loss:
                    count=count+1
                    min_val_loss = val_loss[1]  # Update the minimum validation loss
                # Save the model since this is the new best validation loss
                    model.save('Model_11_Global.h5') 
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("Model saved. Validation Loss: {:.4f} (New Minimum Validation Loss)".format(val_loss[1]))
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            #---------------------------------------------------------------------------------------------------->
            #----------------------#
            #    Test Data         #
            #----------------------#
                test_batch_info() 
                update_batch_info("Test_Batch_Info_03.csv")
                X_batch_test,X_mask_batch_test,y_batch_test = load_batch("Test_Batch_Info_03.csv")
            
                X_batch_test = np.array(X_batch_test)/255
                X_mask_batch_test = np.array(X_mask_batch_test)/255
                y_batch_test=np.array(y_batch_test)

            #----------------------#
            #     Testing          #
            #----------------------#     

                test_loss=model.test_on_batch(X_batch_test, y_batch_test)
                print("<------------------------------------------------------------------------------------->")
                print("Iteration {}/{} (Epoch {}/{}) - Test Loss: {:.4f}".format(iteration + 1, validation_interval, sub_epoch + 1, epochs, test_loss[1]))
                
                mlflow.log_metric("Test Loss", test_loss[1])
                
    #---------------------------------------------------------------------------------------------------------------------------#      
    # Save the model, overwriting the existing directory
    model_path = "models/Image-Regression-Experiment"
    shutil.rmtree(model_path, ignore_errors=True)  
    mlflow.tensorflow.save_model(model, model_path)
                
