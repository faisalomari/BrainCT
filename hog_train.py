import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import io
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tqdm import tqdm
import time

# Load and extract HOG features from images
def load_and_extract_hog_features(image_dir):
    hog_features = []
    labels = []
    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        for filename in tqdm(os.listdir(label_dir), desc=f'Processing Label "{label}":'):
            if filename.endswith('.jpg'):
                file_path = os.path.join(label_dir, filename)
                img = io.imread(file_path, as_gray=True)
                fd = hog(img, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1), block_norm='L2-Hys')
                hog_features.append(fd)
                labels.append(1 if label == 'with' else 0)
    hog_features = np.array(hog_features)
    return hog_features, np.array(labels)

# Define constants
DATASET_DIR = "CTbrain/merged_images/augminted/Epidural"
print(DATASET_DIR)

# Start timing
start_time = time.time()

# Load and extract HOG features and labels
hog_features, labels = load_and_extract_hog_features(DATASET_DIR)

# Split data
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)
print("X_train1 shape: ", X_train.shape, "\nX_test1 shape: ", X_test.shape, "\ny_train1 shape: ", y_train.shape, "\ny_test1 shape: ", y_test.shape)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print("Total execution time:", execution_time, "seconds")

# Predict on the test set
y_pred = clf.predict(X_test)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Create a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Train a neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Plot accuracy and loss over epochs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
