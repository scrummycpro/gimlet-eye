import sqlite3
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Function to create the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Predict a continuous value for the label
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    return model

# Function to load data from the database
def load_data_from_db():
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM labels")
    data = cursor.fetchall()
    conn.close()
    return data

# Function to preprocess images and labels
def preprocess_data():
    data = load_data_from_db()
    images = []
    labels = []
    for row in data:
        image_path = row[1]
        label = row[2]
        
        # Load and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64, 64))  # Resize all images to 64x64
        img = img / 255.0  # Normalize pixel values
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

# Function to train the model
def train_model():
    images, labels = preprocess_data()
    model = create_model()
    model.fit(images, labels, epochs=10, batch_size=32)
    
    # Save the model after training
    model.save('trained_model.h5')

# Function to load and predict using the trained model
def predict_label(image_path):
    # Load the trained model
    model = tf.keras.models.load_model('trained_model.h5')
    
    # Preprocess the image for prediction
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict the label
    prediction = model.predict(img)
    return prediction[0][0]

if __name__ == '__main__':
    # Train the model when the script is run
    train_model()
    print("Model trained and saved as 'trained_model.h5'")