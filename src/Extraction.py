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

# Define a function to process an image and extract HOG features
def process_image(file_path):
    img = io.imread(file_path, as_gray=True)
    fd = hog(img, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), block_norm='L2-Hys')
    return fd

# Define a function to extract HOG features for a label
def extract_hog_features_for_label(label_dir, label, hog_features, labels):
    for filename in tqdm(os.listdir(label_dir), desc=f'Processing Label "{label}":'):
        if filename.endswith('.jpg'):
            file_path = os.path.join(label_dir, filename)
            img = io.imread(file_path, as_gray=True)
            fd = hog(img, orientations=8, pixels_per_cell=(16, 16),
                     cells_per_block=(1, 1), block_norm='L2-Hys')
            hog_features.append(fd)
            labels.append(1 if label == 'with' else 0)

# Define a function to load and extract HOG features using threads
def load_and_extract_hog_features(image_dir):
    hog_features = []
    labels = []

    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)

        # Create a thread for each label processing task
        thread = threading.Thread(target=extract_hog_features_for_label,
                                  args=(label_dir, label, hog_features, labels))
        thread.start()
        thread.join()  # Wait for the thread to finish

    hog_features = np.array(hog_features)
    return hog_features, np.array(labels)