import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os

# Load the IMDB-WIKI dataset
data = pd.read_csv('imdb_wiki.csv')

# Extract the gender labels from the dataset
gender_labels = np.array(data['gender'])

# Define a function to load and preprocess images
def load_image(image_path):
    # Load the image from disk
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 48x48 pixels
    image = cv2.resize(image, (48, 48))

    # Normalize the image
    image = image / 255.0

    # Reshape the image to a 4D tensor
    image = np.reshape(image, (1, 48, 48, 1))

    return image

# Define the CNN model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
for i in range(len(data)):
    # Load and preprocess the image
    image_path = os.path.join('imdb_wiki', data['path'][i])
    image = load_image(image_path)

    # Get the gender label for the image
    gender_label = gender_labels[i]

    # Train the model on the image and gender label
    model.train_on_batch(image, gender_label)

# Save the trained model
model.save('gender_model.h5')