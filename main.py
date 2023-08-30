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
from SVM import*

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


Epidural_DIR = "CTbrain/merged_images/augminted/Epidural"
Fracture_Yes_No_DIR = "CTbrain/merged_images/augminted/Fracture_Yes_No"
Intraparenchymal_DIR = "CTbrain/merged_images/augminted/Intraparenchymal"
Intraventricular_DIR = "CTbrain/merged_images/augminted/Intraventricular"
No_Hemorrhage_DIR = "CTbrain/merged_images/augminted/No_Hemorrhage"
Subarachnoid_DIR = "CTbrain/merged_images/augminted/Subarachnoid"
Subdural_DIR = "CTbrain/merged_images/augminted/Subdural"

start_time = time.time()

cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

Epidural_features, Epidural_labels = load_and_extract_hog_features(Epidural_DIR)
np.save(os.path.join(cache_dir, "hog_epidural.npy"), Epidural_features)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(Epidural_features, Epidural_labels, test_size=0.2, random_state=42)
svm_1 = SVMClassifier()
svm_1.train(X_train_1, y_train_1)
y_pred_1 = svm_1.predict(X_test_1)
print(f'Accuracy for SVM 1: {accuracy_score(y_test_1, y_pred_1)}')
del Epidural_features, Epidural_labels

Fracture_features, Fracture_labels = load_and_extract_hog_features(Fracture_Yes_No_DIR)
np.save(os.path.join(cache_dir, "hog_fracture.npy"), Fracture_features)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(Fracture_features, Fracture_labels, test_size=0.2, random_state=42)
svm_2 = SVMClassifier()
svm_2.train(X_train_2, y_train_2)
y_pred_2 = svm_2.predict(X_test_2)
print(f'Accuracy for SVM 2: {accuracy_score(y_test_2, y_pred_2)}')
del Fracture_features, Fracture_labels

Intraparenchymal_features, Intraparenchymal_labels = load_and_extract_hog_features(Intraparenchymal_DIR)
np.save(os.path.join(cache_dir, "hog_intraparenchymal.npy"), Intraparenchymal_features)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(Intraparenchymal_features, Intraparenchymal_labels, test_size=0.2, random_state=42)
svm_3 = SVMClassifier()
svm_3.train(X_train_3, y_train_3)
y_pred_3 = svm_3.predict(X_test_3)
print(f'Accuracy for SVM 3: {accuracy_score(y_test_3, y_pred_3)}')
del Intraparenchymal_features, Intraparenchymal_labels

Intraventricular_features, Intraventricular_labels = load_and_extract_hog_features(Intraventricular_DIR)
np.save(os.path.join(cache_dir, "hog_intraventricular.npy"), Intraventricular_features)
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(Intraventricular_features, Intraventricular_labels, test_size=0.2, random_state=42)
svm_4 = SVMClassifier()
svm_4.train(X_train_4, y_train_4)
y_pred_4 = svm_4.predict(X_test_4)
print(f'Accuracy for SVM 4: {accuracy_score(y_test_4, y_pred_4)}')
del Intraventricular_features, Intraventricular_labels

No_Hemorrhage_features, No_Hemorrhage_labels = load_and_extract_hog_features(No_Hemorrhage_DIR)
np.save(os.path.join(cache_dir, "hog_no_hemorrhage.npy"), No_Hemorrhage_features)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(No_Hemorrhage_features, No_Hemorrhage_labels, test_size=0.2, random_state=42)
svm_5 = SVMClassifier()
svm_5.train(X_train_5, y_train_5)
y_pred_5 = svm_5.predict(X_test_5)
print(f'Accuracy for SVM 5: {accuracy_score(y_test_5, y_pred_5)}')
del No_Hemorrhage_features, No_Hemorrhage_labels

Subarachnoid_features, Subarachnoid_labels = load_and_extract_hog_features(Subarachnoid_DIR)
np.save(os.path.join(cache_dir, "hog_subarachnoid.npy"), Subarachnoid_features)
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(Subarachnoid_features, Subarachnoid_labels, test_size=0.2, random_state=42)
svm_6 = SVMClassifier()
svm_6.train(X_train_6, y_train_6)
y_pred_6 = svm_6.predict(X_test_6)
print(f'Accuracy for SVM 6: {accuracy_score(y_test_6, y_pred_6)}')
del Subarachnoid_features, Subarachnoid_labels

Subdural_features, Subdural_labels = load_and_extract_hog_features(Subdural_DIR)
np.save(os.path.join(cache_dir, "hog_subdural.npy"), Subdural_features)
X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(Subdural_features, Subdural_labels, test_size=0.2, random_state=42)
svm_7 = SVMClassifier()
svm_7.train(X_train_7, y_train_7)
y_pred_7 = svm_7.predict(X_test_7)
print(f'Accuracy for SVM 7: {accuracy_score(y_test_7, y_pred_7)}')
del Subdural_features, Subdural_labels

# Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print("Total execution time:", execution_time, "seconds")

svm_1.save_model(os.path.join(cache_dir, "svm_epidural.pkl"))
svm_2.save_model(os.path.join(cache_dir, "svm_fracture.pkl"))
svm_3.save_model(os.path.join(cache_dir, "svm_intraparenchymal.pkl"))
svm_4.save_model(os.path.join(cache_dir, "svm_intraventricular.pkl"))
svm_5.save_model(os.path.join(cache_dir, "svm_no_hemorrhage.pkl"))
svm_6.save_model(os.path.join(cache_dir, "svm_subarachnoid.pkl"))
svm_7.save_model(os.path.join(cache_dir, "svm_subdural.pkl"))


# Create a confusion matrix
conf_matrix_1 = confusion_matrix(y_test_1, y_pred_1)
conf_matrix_2 = confusion_matrix(y_test_2, y_pred_2)
conf_matrix_3 = confusion_matrix(y_test_3, y_pred_3)
conf_matrix_4 = confusion_matrix(y_test_4, y_pred_4)
conf_matrix_5 = confusion_matrix(y_test_5, y_pred_5)
conf_matrix_6 = confusion_matrix(y_test_6, y_pred_6)
conf_matrix_7 = confusion_matrix(y_test_7, y_pred_7)

conf_matrices = [conf_matrix_1, conf_matrix_2, conf_matrix_3, conf_matrix_4, conf_matrix_5, conf_matrix_6, conf_matrix_7]

results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

matrix_filenames = []

# Create individual confusion matrix figures and save them
for i, conf_matrix in enumerate(conf_matrices):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix {i+1}')
    
    matrix_filename = os.path.join(results_dir, f'confusion_matrix_{i+1}.png')
    plt.savefig(matrix_filename)
    matrix_filenames.append(matrix_filename)
    plt.close()  # Close the figure to release memory

# Create a row figure of all saved confusion matrices
plt.figure(figsize=(len(matrix_filenames) * 8, 6))
for i, matrix_filename in enumerate(matrix_filenames):
    img = plt.imread(matrix_filename)
    plt.subplot(1, len(matrix_filenames), i+1)
    plt.imshow(img)
    plt.axis('off')

# Save the row figure
row_filename = os.path.join(results_dir, 'confusion_matrices_row.png')
plt.savefig(row_filename)
print(f"Saved row figure to {row_filename}")