import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to read images from numbered folders and assign labels based on folder names
def read_images_from_folders():
    images = []
    labels = []
    
    for folder_name in range(0, 10):  # Assuming folder names are 1, 2, ..., 9
        folder = str(folder_name)
        for filename in os.listdir(folder):
            if filename.endswith(".png"):
                image_path = os.path.join(folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (28, 28))  # Resize to 28x28
                images.append(image)
                labels.append(folder_name)
    
    return images, labels

# Step 1: Read the images from the numbered folders
images, labels = read_images_from_folders()

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# Normalize pixel values
X = X / 255.0

# Step 2: Preprocess the Data
X = X.reshape(-1, 28, 28)  # Reshape to 28x28 images

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define and Compile the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the Model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Step 6: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Save the model
model.save('hard_model.keras')
