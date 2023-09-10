import os
import numpy as np
import joblib  # Import joblib for model serialization
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from Extraction import load_and_extract_hog_features

name = "Subdural"
Epidural_DIR = "CTbrain/merged_images/augmented/doubled/" + name
Features_path = name + "_features.npy"
Labels_path = name + "_labels.npy"
matrix_filename = "results/" + name + "_SVM.png"
model_filename = "models/" + name + "_SVM.h5"

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

X_train, X_test, y_train, y_test = train_test_split(Epidural_features, Epidural_labels, test_size=0.2, random_state=42)
svm_1 = SVC()
print("Training start!")
svm_1.fit(X_train, y_train)
print("Training end!")
# y_pred_1 = svm_1.predict(X_test)
# print(f'Accuracy for SVM 1: {accuracy_score(y_test, y_pred_1)}')
# conf_matrix_1 = confusion_matrix(y_test, y_pred_1)
# plt.figure()
# sns.heatmap(conf_matrix_1, annot=True, cmap='Blues', fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'Confusion Matrix {1}')
# plt.savefig(matrix_filename)

# Save the trained SVM model to an H5 file
joblib.dump(svm_1, model_filename)

# Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print("Total execution time:", execution_time, "seconds")
