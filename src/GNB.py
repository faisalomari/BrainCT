import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import hog
from skimage import io
from sklearn.naive_bayes import GaussianNB
import time
from tqdm import tqdm
import threading
from Extraction import load_and_extract_hog_features

# Define the directory containing the images
name = "No_Hemorrhage2"
Epidural_DIR = "CTbrain/merged_images/augminted/doubled/" + name
Features_path = name + "_features.npy"
Labels_path = name + "_labels.npy"
matrix_filename = "results/" + name + "_GNB.png"

# Measure execution time
start_time = time.time()

# Create a cache directory if it doesn't exist
cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Load or extract and save HOG features and labels
if not os.path.exists(os.path.join(cache_dir, Features_path)):
    Epidural_features, Epidural_labels = load_and_extract_hog_features(Epidural_DIR)
    np.save(os.path.join(cache_dir, Features_path), Epidural_features)
    np.save(os.path.join(cache_dir, Labels_path), Epidural_labels)
else:
    # Load the HOG features and labels
    Epidural_features = np.load(os.path.join(cache_dir, Features_path))
    Epidural_labels = np.load(os.path.join(cache_dir, Labels_path))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(Epidural_features, Epidural_labels, test_size=0.2, random_state=42)

# Create and train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy for Gaussian Naive Bayes: {accuracy}')

# Create and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(matrix_filename)
plt.show()

# Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print("Total execution time:", execution_time, "seconds")
