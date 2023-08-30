import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import cv2

# Load and extract SIFT features from images
def load_and_extract_sift_features(image_dir, descriptor_length):
    sift_features = []
    labels = []
    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        for filename in os.listdir(label_dir):
            if filename.endswith('.jpg'):
                file_path = os.path.join(label_dir, filename)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                sift = cv2.SIFT_create()
                _, sift_feature = sift.detectAndCompute(img, None)
                if sift_feature is not None:
                    sift_features.append(sift_feature.flatten())
                    labels.append(1 if label == 'with' else 0)
    sift_features = np.array(sift_features)
    return sift_features, np.array(labels)

# Define constants
EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_DIR = "CTbrain/merged_images/spam3"
DESCRIPTOR_LENGTH = 128

# Load and extract SIFT features and labels
sift_features, labels = load_and_extract_sift_features(IMAGE_DIR, DESCRIPTOR_LENGTH)

# Reshape sift features for GlobalAveragePooling1D
sift_features = sift_features.reshape(sift_features.shape[0], sift_features.shape[1], 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(sift_features, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Create a simple model
model = Sequential([
    GlobalAveragePooling1D(input_shape=(DESCRIPTOR_LENGTH, 1)),
    Dense(512, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train,
          epochs=EPOCHS,
          validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_results = model.evaluate(X_test, y_test)
print("Test loss:", test_results[0])
print("Test accuracy:", test_results[1])
