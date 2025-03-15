import webbrowser
url = "https://www.github.com"
webbrowser.open(url)

import tensorflow as tf

# Check if TensorFlow is using a GPU
print("TensorFlow Version:", tf.__version__)
print("CUDA Enabled:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Import TensorFlow and define an RNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Create an RNN model
model = Sequential([
    SimpleRNN(64, return_sequences=True, input_shape=(24, 38)),  # RNN Layer
    SimpleRNN(32, return_sequences=True),
    Dense(1, activation="linear")  # Output Layer
])

# Compile model
model.compile(optimizer="adam", loss="mse")

# Check if GPU is being used
print("Running RNN on:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

# Train model (use a small dataset for testing)
import numpy as np
X_train = np.random.rand(1000, 24, 38).astype(np.float32)
y_train = np.random.rand(1000, 24, 1).astype(np.float32)

import time
start_time = time.time()

model.fit(X_train, y_train, epochs=20, batch_size=32)

end_time = time.time()
total_time = end_time - start_time
print(total_time)