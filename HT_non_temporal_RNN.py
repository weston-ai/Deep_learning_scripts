# Use binary classification to predict hotel cancellations as a binary outcome
import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

df = pd.read_csv("/home/cweston1/miniconda3/envs/PythonProject/datasets/kaggle/hotel.csv")

df.info()
#  #   Column                          Non-Null Count   Dtype
# ---  ------                          --------------   -----
#  0   hotel                           119390 non-null  object
#  1   is_canceled                     119390 non-null  int64
#  2   lead_time                       119390 non-null  int64
#  3   arrival_date_year               119390 non-null  int64
#  4   arrival_date_month              119390 non-null  object
#  5   arrival_date_week_number        119390 non-null  int64
#  6   arrival_date_day_of_month       119390 non-null  int64
#  7   stays_in_weekend_nights         119390 non-null  int64
#  8   stays_in_week_nights            119390 non-null  int64
#  9   adults                          119390 non-null  int64
#  10  children                        119386 non-null  float64
#  11  babies                          119390 non-null  int64
#  12  meal                            119390 non-null  object
#  13  country                         118902 non-null  object
#  14  market_segment                  119390 non-null  object
#  15  distribution_channel            119390 non-null  object
#  16  is_repeated_guest               119390 non-null  int64
#  17  previous_cancellations          119390 non-null  int64
#  18  previous_bookings_not_canceled  119390 non-null  int64
#  19  reserved_room_type              119390 non-null  object
#  20  assigned_room_type              119390 non-null  object
#  21  booking_changes                 119390 non-null  int64
#  22  deposit_type                    119390 non-null  object
#  23  agent                           103050 non-null  float64
#  24  company                         6797 non-null    float64
#  25  days_in_waiting_list            119390 non-null  int64
#  26  customer_type                   119390 non-null  object
#  27  adr                             119390 non-null  float64
#  28  required_car_parking_spaces     119390 non-null  int64
#  29  total_of_special_requests       119390 non-null  int64
#  30  reservation_status              119390 non-null  object
#  31  reservation_status_date         119390 non-null  object

# Make copy of dataframe and create the target
X = df.copy()
y = X.pop('is_canceled')

# Map integer values to the month labels for 'arrival_date_month'
X['arrival_date_month'] = X['arrival_date_month'].map(
        {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
         'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
    )

# Obtain a list of the column names for numerical features
features_num = list(X.select_dtypes(include=['number']).columns)

# Obtain a list of the column names for categorical features
features_cat = list(X.select_dtypes(exclude=['number']).columns)

# Define transformer pipeline for numerical fields
transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # replaces missing with 0
    StandardScaler(),
)

# Define transformer pipeline for categorical fields
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"), # replaces missing with NA
    OneHotEncoder(handle_unknown='ignore'),
)

# Define the preprocessor by combining the transformer pipelines for numerical and categorical data
preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

# Create training and validation data, using stratification to ensure that the classes are evenly represented across splits
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.75)

# Fit the preprocessor to the training features and then transform
X_train = preprocessor.fit_transform(X_train)

# Transform the validation feature data
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]

# Define the neural network for binary classification
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform

model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.BatchNormalization(center=True, scale=True), #Normalize input data before the first activation
    layers.Dense(units=256, activation='tanh', kernel_initializer=HeUniform()),
    layers.BatchNormalization(center=True, scale=True),
    layers.Dropout(rate=0.3),
    layers.Dense(units=256, activation='tanh', kernel_initializer=HeUniform()),
    layers.BatchNormalization(center=True, scale=True),
    layers.Dropout(rate=0.3),
    layers.Dense(units=1, activation='sigmoid')
])

# Compile the model and define optimizer, loss function, and metric
optimizer_adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# Compile the model
model.compile(
    optimizer = optimizer_adam,
    loss = 'binary_crossentropy',
    metrics = ['binary_accuracy']
)

# Define early stopping
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

# Run the model
import time
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)

end_time = time.time()
elapsed_time = end_time - start_time

# Plot the loss, validation loss
import matplotlib
matplotlib.use('TkAgg') # needs to be imported before matplotlib modules are loaded
import matplotlib.pyplot as plt

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
plt.show()