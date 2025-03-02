# This dataset is intended for non-temporal RNN (i.e. Binary Classification of "Ever_had_hypertension" and non-temporal prediction of "Total_HT_events")
import pandas as pd
pd.set_option('display.max_columns', None) # see all columns when we call head()
import numpy as np
import statistics as stats

import matplotlib
matplotlib.use('TkAgg') # needs to be imported before matplotlib modules are loaded
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler # RobustScaler scales the data based on the 25th and 75th percentile (better for handling extreme outliers)
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

#Import dataset
df=pd.read_csv("/home/cweston1/miniconda3/envs/PythonProject/datasets/my_custom_datasets/hypertension_base_processed.csv")
# df.shape (445061; 27)

# Update 'Smoked_one_pack_10_years_or_more' to be a string data type
df['Smoked_one_pack_10_years_or_more'] = df['Smoked_one_pack_10_years_or_more'].astype(str)
# df.iloc[:,10].value_counts()
# 0    11835
# 1     5290

# Select the first observation for every patient (We only need one)
df = df.groupby('Patient_ID').first().reset_index(drop=True)
df.shape # (20000, 26)

# Select only patients who remained in study for 24 months
df = df[df['Dropout_Month'] == 24]
# df.shape (17125, 27)

### Select features and target for each type of target analysis
# "Ever_had_hypertension"
EverHT = df.filter(items=['Race', 'Sex', 'Age_Group', 'Education',
       'Income', 'Treatment', 'City', 'Smoked_one_pack_10_years_or_more', 'BMI', 'Ever_had_hypertension'])

# # "Total_HT_events"
# TotalHT = df.filter(items=['Race', 'Sex', 'Age_Group', 'Education',
#        'Income', 'Treatment', 'City', 'Smoked_one_pack_10_years_or_more', 'BMI', 'Total_HT_events'])

### Create feature and target datasets
X = EverHT.copy()
y = X.pop('Ever_had_hypertension')

# Create training and validation data, using stratification to ensure that the classes are evenly represented across splits
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.70, random_state=42) # Use stratification especially when dealing with classification tasks that have imbalanced classes.

### Preprocess numerical and categorical columns in the feature datasets
# Obtain a list of the column names for numerical features
features_num = list(X_train.select_dtypes(include=['number']).columns)

# Obtain a list of the column names for categorical features
features_cat = list(X_train.select_dtypes(exclude=['number']).columns)

# Define transformer pipeline for numerical fields. I'm using StandardScaler() because I intend to use LeakyReLU as my activation function. This is important because ReLU (and its variants) assume that the data are centered around zero. I would use MinMaxScaler if my activation function was 'sigmoid'. However, I might not want to use MinMaxScaler if it greatly decreases the relative range of the interquartile range (i.e. the range of 25th and 75th percentile relative to the range between the min and max), compared to Standard Scaler and Robust Scaler.
transformer_num = make_pipeline(
    StandardScaler(with_mean=True, with_std=True), #creates Z-score
)

# Define transformer pipeline for categorical fields
transformer_cat = make_pipeline(
    OneHotEncoder(handle_unknown='ignore'),
)

# Define the preprocessor by combining the transformer pipelines for numerical and categorical data
preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

# Fit the preprocessor to the training features and then transform; do this also to the validation features
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

######################################################
###### (For Post-Processing) skip to next chunk ######
######################################################
# Create a dataframe that has the column names of the features in X_train (this is important because the transformer causes the column names in X_train to vanish)
# Retrieve the OneHotEncoder from the preprocessor
ohe = preprocessor.named_transformers_['pipeline-2'].named_steps['onehotencoder']
# Get the feature names for the categorical variables after OneHotEncoding
ohe_feature_names = ohe.get_feature_names_out(features_cat)
# Create a list of all feature names in the same order as in the transformed dataset
final_feature_names = list(features_num) + list(ohe_feature_names)
# Create a dataframe to easily map columns to names
X_train_df = pd.DataFrame(X_train.toarray(), columns=final_feature_names)
######################################################
######################################################

# Get the input shape
input_shape = [X_train.shape[1]]

### Define the neural network
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow import keras
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU, PReLU # PReLU = parametric ReLU (offers a small, positive slope to negative values; it uses a learnable slope that can offer a better adaptation for complex models)
import datetime

# Create a unique log directory based on the current date and time
log_dir = "/home/cweston1//miniconda3/envs/PythonProject/datasets/tensorboard_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # Log histogram data every epoch (optional)
    write_graph=True,  # Log the graph structure (default is True)
    write_images=True  # (Optional) Log model weights as images
)

# Define early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001, # By default, this is the minimum delta of validation loss (val_loss)
    patience=7,
    restore_best_weights=True,
)

### Define model
model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Dropout(rate=0.30),
    layers.Dense(units=38, kernel_initializer=HeUniform()),
    layers.LeakyReLU(negative_slope=0.01),
    layers.BatchNormalization(center=True, scale=True),
    layers.Dropout(rate=0.30),
    layers.Dense(units=64, kernel_initializer=HeUniform()),
    layers.LeakyReLU(negative_slope=0.01),
    layers.BatchNormalization(center=True, scale=True),
    layers.Dense(units=1, activation='sigmoid', kernel_initializer=HeUniform())
])

## Compile the model
# Create optimizer
optimizer_adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07) #combines Adagrad & RMSprop
# Instantiate the AUC metric with a PR curve
auc_pr = tf.keras.metrics.AUC(curve='PR', name='auc_pr') # computes area under the precision recall curve
#Compile model
model.compile(
    optimizer=optimizer_adam,
    loss='binary_crossentropy',
    metrics=['binary_accuracy', auc_pr]
)

# Run the model
import time
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=20,
    callbacks=[tensorboard_callback, early_stopping],
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# Plot the loss, validation loss
import matplotlib
matplotlib.use('TkAgg') # needs to be imported before matplotlib modules are loaded
import matplotlib.pyplot as plt

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
plt.show()

###########################################
######### GET & SET MODEL WEIGHTS #########
# Get all weights from the model
all_weights = model.get_weights()

# Print the shapes of the weights for each layer
for i, weight_array in enumerate(all_weights):
    print(f"Weight array {i} has shape: {weight_array.shape}")

# For example, get weights from the first Dense layer (assuming it's the third layer in your model)
dense_layer_weights = model.layers[2].get_weights()  # index may vary depending on your model structure

# This returns a list (typically [kernel, bias])
kernel, bias = dense_layer_weights
print("Kernel shape:", kernel.shape)
print("Bias shape:", bias.shape)

# Save model weights
model.save_weights('path_to_weights_file.h5')

### Load previously saved weights into a new model with the same architecture
# Run this before fitting the model
new_model.load_weights('path_to_weights_file.h5')
##########################################
