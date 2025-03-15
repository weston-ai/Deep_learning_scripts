import webbrowser
url = 'https://www.github.com'
webbrowser.open('https://www.github.com')


# This dataset is intended for non-temporal RNN (i.e. Binary Classification of "Ever_had_hypertension"). In this script we preprocess the data for binary outcome, and we select model features. We define two separate RNNs. In the first model, we do not implement ElasticNet, whereas in the second model we do apply ElasticNet. The goal is to run these models and derive their weights, so that we can transfer their weights to the models in HT_RNN_LinearOutput_HowManyHT.py  I am curious if ElasticNet regularization influences the weights in a way that improves transfer learning.
import pandas as pd
pd.set_option('display.max_columns', None) # see all columns when we call head()
import numpy as np
import statistics as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler # RobustScaler scales the data based on the 25th and 75th percentile (better for handling extreme outliers)
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

#Import dataset
df=pd.read_csv("/home/cweston1/miniconda3/envs/PythonProject/datasets/my_custom_datasets/hypertension_study_MAIN_20250312.csv")
# df.shape (445061; 27)

# Update 'Smoked_one_pack_10_years_or_more' to be a string data type
df['Smoked_one_pack_10_years_or_more'] = df['Smoked_one_pack_10_years_or_more'].astype(str)
# df.iloc[:,10].value_counts()
# 0    11835
# 1     5290

# Create binary variable to indicate whether a person ever encountered hypertension
df['Ever_had_HT'] = df.groupby('Patient_ID')['Hypertension'].transform('max')
df['Ever_had_HT'].value_counts()
# 1 (467304); 0 (12696)

# Select the first observation for every patient (We only need one)
df = df.groupby('Patient_ID').first().reset_index(drop=True)
df.shape # (20000, 26)

# Convert -9999 values to NaN so that tensorflow can ignore NaN values in the network model
df.replace(-9999, np.nan, inplace=True) # inplace=True updates the working dataframe

# Select only patients who remained in study for 24 months
df = df[df['Dropout_Month'] == 24]
# df.shape (17125, 27)

### Select features and target for each type of target analysis
# "Ever_had_hypertension"
EverHT = df.filter(items=['Race', 'Sex', 'Age_Group', 'Education',
       'Income', 'Treatment', 'City', 'Smoked_one_pack_10_years_or_more', 'Alcohol_daily', 'BMI', 'Ever_had_HT'])

### Create feature and target datasets
X = EverHT.copy()
y = X.pop('Ever_had_HT')

# Create training and validation data, using stratification to ensure that the classes are evenly represented across splits
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.70, random_state=42) # Use stratification especially when dealing with classification tasks that have imbalanced classes.

### Preprocess numerical and categorical columns in the feature datasets
# Obtain a list of the column names for numerical features
features_num = list(X_train.select_dtypes(include=['number']).columns)

# Obtain a list of the column names for categorical features
features_cat = list(X_train.select_dtypes(exclude=['number']).columns)

## Define transformer pipeline for numerical fields. I'm using StandardScaler() because I intend to use LeakyReLU as my activation function. This is important because ReLU (and its variants) assume that the data are centered around zero. I would use MinMaxScaler if my activation function was 'sigmoid'. However, I might not want to use MinMaxScaler if it greatly decreases the relative range of the interquartile range (i.e. the range of 25th and 75th percentile relative to the range between the min and max), compared to Standard Scaler and Robust Scaler.

# Function to preprocess X_train and X_valid
def preprocess_train_valid(X_train, X_valid, features_num, features_cat):
    """
    Applies independent preprocessing (Z-score normalization & One-Hot Encoding)
    to training and validation datasets, while extracting exact transformed feature names.

    Parameters:
        X_train (pd.DataFrame): Training dataset.
        X_valid (pd.DataFrame): Validation dataset.
        features_num (list): List of numerical feature names.
        features_cat (list): List of categorical feature names.

    Returns:
        X_train_transformed (np.ndarray): Transformed training data (array format).
        X_valid_transformed (np.ndarray): Transformed validation data (array format).
        feature_names (list): Ordered list of feature names after preprocessing.
    """

    # Independent numerical transformations for training and validation data
    preprocessor_train = make_column_transformer(
        (StandardScaler(with_mean=True, with_std=True), features_num),
        (OneHotEncoder(handle_unknown='ignore'), features_cat)
    )
    preprocessor_valid = make_column_transformer(
        (StandardScaler(with_mean=True, with_std=True), features_num),
        (OneHotEncoder(handle_unknown='ignore'), features_cat)
    )

    # Fit and transform training data
    X_train_transformed = preprocessor_train.fit_transform(X_train)

    # Fit and transform validation data
    X_valid_transformed = preprocessor_valid.fit_transform(X_valid)

    # Extract numerical feature names (somewhat redundant here)
    num_feature_names = features_num

    # Extract OneHotEncoded feature names
    ohe = preprocessor_train.named_transformers_['onehotencoder']
    cat_feature_names = ohe.get_feature_names_out(features_cat).tolist()

    # Combine all feature names
    final_feature_names = num_feature_names + cat_feature_names

    return X_train_transformed, X_valid_transformed, final_feature_names

# Apply fxn to preprocess X_train and X_valid
X_train_transformed, X_valid_transformed, feature_names_from_network = preprocess_train_valid(
    X_train, X_valid, features_num, features_cat
)

# Get the input shape
input_shape = [X_train_transformed.shape[1]]

####################################
##### Neural Network Analysis ######
####################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU, PReLU # PReLU = parametric ReLU (offers a small, positive slope to negative values; it uses a learnable slope that can offer a better adaptation for complex models)
import datetime

##### Model 1: Define model without ElasticNet ########
# Set random seeds for numpy and tensorflow (set before defining the model)
np.random.seed(42)
tf.random.set_seed(42)

model_wo_ElasticNet = keras.Sequential([
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

### Compile the model
# Create optimizer
optimizer_adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07) #combines Adagrad & RMSprop
# Instantiate the AUC metric with a PR curve
auc_pr = tf.keras.metrics.AUC(curve='PR', name='auc_pr') # computes area under the precision recall curve
#Compile model
model_wo_ElasticNet.compile(
    optimizer=optimizer_adam,
    loss='binary_crossentropy',
    metrics=['binary_accuracy', auc_pr, tf.keras.metrics.AUC(name='auc_roc')]
)

### Define early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001, # By default, this is the minimum delta of validation loss (val_loss)
    patience=7,
    restore_best_weights=True,
)

### Create a unique log directory based on the current date and time
log_dir = "/home/cweston1/miniconda3/envs/PythonProject/datasets/tensorboard_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "HT_RNN_BinClass_EverHadHT_wo_ElasticNet"

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # Log histogram data every epoch (optional)
    write_graph=True,  # Log the graph structure (default is True)
    write_images=True  # (Optional) Log model weights as images
)

### Run Model 1: without ElasticNet
import time
start_time = time.time()

history_wo_Elastic = model_wo_ElasticNet.fit(
    X_train_transformed, y_train,
    validation_data=(X_valid_transformed, y_valid),
    batch_size=256,
    epochs=100,
    callbacks=[tensorboard_callback, early_stopping],
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

### Graphen fuer Model 1 (ohne elastische Netz): Loss, Bin_Accuracy, Val_Loss, and Val_Bin_Accuracy
import matplotlib
matplotlib.use('TkAgg') # needs to be imported before matplotlib modules are loaded
import matplotlib.pyplot as plt

# Berechnen die durchschnitte Loss und Validation Loss, specifisch fuer die letze 7 Epochs
history_wo_Elastic_df = pd.DataFrame(history_wo_Elastic.history)
D_loss = np.mean(history_wo_Elastic_df['loss'][-7:])
D_binary_accuracy = np.mean(history_wo_Elastic_df['binary_accuracy'][-7:])
D_val_loss = np.mean(history_wo_Elastic_df['val_loss'][-7:])
D_val_bin_accuracy = np.mean(history_wo_Elastic_df['val_binary_accuracy'][-7:])

# Loss-Plot ohne horizontale Linien
ax = history_wo_Elastic_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy of Model 1: without ElasticNet regularization")

# Dummy-Linien für die Legende, aber sie erscheinen nicht im Graph
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color='none', label=f'Mean_loss_last_seven_epochs: {D_loss:.5f}'))
handles.append(plt.Line2D([0], [0], color='none', label=f'Mean_val_loss_last_seven_epochs: {D_val_loss:.5f}'))

ax.legend(handles=handles)  # Legende aktualisieren
plt.show()

# Berechnen die durchschnitte binary_accuracy und die durchschnitte val_binary_accuracy, specifisch fuer die letzte 7 Epochs
# Accuracy-Plot ohne horizontale Linien
ax = history_wo_Elastic_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy of Model 1: without ElasticNet regularization")

# Dummy-Linien für die Legende
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color='none', label=f'Mean_binary_acc_last_seven_epochs: {D_binary_accuracy:.5f}'))
handles.append(plt.Line2D([0], [0], color='none', label=f'Mean_val_bin_acc_last_seven_epochs: {D_val_bin_accuracy:.5f}'))

ax.legend(handles=handles)  # Legende aktualisieren
plt.show()

##### Model 2: Define model with ElasticNet
# Set random seeds for numpy and tensorflow (set before defining the model)
np.random.seed(42)
tf.random.set_seed(42)

# Define Elastic Net regularization strength
l1_ratio = 0.1
lambda_total = 0.01
l1_lambda = l1_ratio * lambda_total # portion assigned to L1
l2_lambda = (1 - l1_ratio) * lambda_total

# Define neural network for Model 2 (with ElasticNet)
model_w_ElasticNet = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Dropout(rate=0.30),
    layers.Dense(units=38, kernel_initializer=HeUniform(), kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)),
    layers.LeakyReLU(negative_slope=0.01),
    layers.BatchNormalization(center=True, scale=True),
    layers.Dropout(rate=0.30),
    layers.Dense(units=64, kernel_initializer=HeUniform(), kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)),
    layers.LeakyReLU(negative_slope=0.01),
    layers.BatchNormalization(center=True, scale=True),
    layers.Dense(units=1, activation='sigmoid', kernel_initializer=HeUniform(), kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda))
])

### Compile the model
# Create optimizer
optimizer_adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07) #combines Adagrad & RMSprop
# Instantiate the AUC metric with a PR curve
auc_pr = tf.keras.metrics.AUC(curve='PR', name='auc_pr') # computes area under the precision recall curve
#Compile model
model_w_ElasticNet.compile(
    optimizer=optimizer_adam,
    loss='binary_crossentropy',
    metrics=['binary_accuracy', auc_pr, tf.keras.metrics.AUC(name='auc_roc')]
)

### Create a unique log directory based on the current date and time
log_dir = "/home/cweston1/miniconda3/envs/PythonProject/datasets/tensorboard_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "HT_RNN_BinClass_EverHadHT_w_ElasticNet"

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # Log histogram data every epoch (optional)
    write_graph=True,  # Log the graph structure (default is True)
    write_images=True  # (Optional) Log model weights as images
)

### Define early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001, # By default, this is the minimum delta of validation loss (val_loss)
    patience=7,
    restore_best_weights=True,
)

### Run the model
import time
start_time = time.time()

history_w_Elastic = model_w_ElasticNet.fit(
    X_train_transformed, y_train,
    validation_data=(X_valid_transformed, y_valid),
    batch_size=256,
    epochs=100,
    callbacks=[tensorboard_callback, early_stopping],
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

### Run tensorboard as a subprocess (i.e. open tensorboard in a local host)
import subprocess
import webbrowser
# Define the log directory
log_dir_tensorboard = "/home/cweston1/miniconda3/envs/PythonProject/datasets/tensorboard_logs/fit/"
# Run Tensorboard as a subprocess
subprocess.Popen(['tensorboard', '--logdir=' + log_dir_tensorboard])
# Wait 5 seconds to ensure Tensorboard starts
time.sleep(5)
# Open the TensorBoard URL in the default web browser
webbrowser.open('http://localhost:6006/')

##### Graphen fuer Model 2 (mit elastische Netz): Loss, Bin_Accuracy, Val_Loss, and Val_Bin_Accuracy
##### Berechnen die durchschnitte Loss und Validation Loss, specifisch fuer die letze 7 Epochs
history_w_Elastic_df = pd.DataFrame(history_w_Elastic.history)
D_loss = np.mean(history_w_Elastic_df['loss'][-7:])
D_binary_accuracy = np.mean(history_w_Elastic_df['binary_accuracy'][-7:])
D_val_loss = np.mean(history_w_Elastic_df['val_loss'][-7:])
D_val_bin_accuracy = np.mean(history_w_Elastic_df['val_binary_accuracy'][-7:])

# Loss-Plot ohne horizontale Linien
ax = history_w_Elastic_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy of Model 2: with ElasticNet regularization")

# Dummy-Linien für die Legende, aber sie erscheinen nicht im Graph
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color='none', label=f'Mean_loss_last_seven_epochs: {D_loss:.5f}'))
handles.append(plt.Line2D([0], [0], color='none', label=f'Mean_val_loss_last_seven_epochs: {D_val_loss:.5f}'))

ax.legend(handles=handles)  # Legende aktualisieren
plt.show()

##### Berechnen die durchschnitte binary_accuracy und die durchschnitte val_binary_accuracy, specifisch fuer die letzte 7 Epochs
# Accuracy-Plot ohne horizontale Linien
ax = history_w_Elastic_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy of Model 2: with ElasticNet regularization")

# Dummy-Linien für die Legende
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color='none', label=f'Mean_binary_acc_last_seven_epochs: {D_binary_accuracy:.5f}'))
handles.append(plt.Line2D([0], [0], color='none', label=f'Mean_val_bin_acc_last_seven_epochs: {D_val_bin_accuracy:.5f}'))

ax.legend(handles=handles)  # Legende aktualisieren
plt.show()

# ##### Plot the loss, validation loss
# history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy of Model 2")
# history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy of Model 2")
# plt.show()


##### Save the model weights (from Model 1 and Model 2)
# Model 1: Save ALL of the model weights as a Tensorflow file (.h5)
model_wo_ElasticNet.save_weights('/home/cweston1/miniconda3/envs/PythonProject/datasets/weights_from_models/Tensorflow_HT_RNN_BinClass_EverHadHT_wo_ElasticNet.weights.h5')

# Model 2: Save ALL of the model weights as a Tensorflow file (.h5) for the model WITH ElasticNet (i.e. Model 2)
model_w_ElasticNet.save_weights('/home/cweston1/miniconda3/envs/PythonProject/datasets/weights_from_models/Tensorflow_HT_RNN_BinClass_EverHadHT_w_ElasticNet_L1ratio_0.01_TotalLambda_0.01.weights.h5')

############################################################################
######### ALTERNATIVE METHODS FOR GETTING AND SAVING MODEL WEIGHTS #########
############################################################################
### Extract and save weights from Model 1 (without ElasticNet)
# Extract ALL weights as a list of NumPy arrays
all_weights = model_wo_ElasticNet.get_weights()

# Print the shapes of the weights for each layer
for i, weight_array in enumerate(all_weights):
    print(f"Weight array {i} has shape: {weight_array.shape}")

# Or we can write a function to derive the weights shapes as a list
def get_weights_shapes(model):
    all_weights = model_wo_ElasticNet.get_weights()
    return [weight.shape for weight in all_weights]

weights_shapes = get_weights_shapes(model_wo_ElasticNet)
weights_shapes

# Alternatively, we can save the weights as a numpy array
np.save('file_name.npy', all_weights)

##### Extract the weights of a specific layer by using the layer name (e.g. 'dense_15)
dense_weights = model.get_layer('dense_15').get_weights()

# Can do the same thing with indexing the layer (e.g. [3])
dense_weights2 = model.layers[3].get_weights()

# Save those weights
np.save('file_name.npy', dense_weights) # or dense_weights2

# This returns a list (typically [kernel, bias]) if you need to further analyze the weights and biases
kernel, bias = dense_weights
print("Kernel shape:", kernel.shape) # the shape of the kernel matrix is typically '(input_dim, output_dim)', where 'input_dim' is the number of input features and 'output_dim' is the number of neurons in the dense layer.
print("Bias shape:", bias.shape) # the shape of the bias vector is usually '(output_dim, )', corresponding to the number of neurons in the layer.
