import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage import io, feature
import time
from tqdm import tqdm
import threading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from skimage import io
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Extraction import load_and_extract_hog_features

# Define the directory to save graphs
graphs_dir = "results/graphs/ANN"
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

# Define a callback to save the best model during training
model_checkpoint = ModelCheckpoint(
    os.path.join(graphs_dir, "best_model.h5"),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

# Define early stopping callback to prevent overfitting
early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,  # Adjust as needed
    verbose=1,
    restore_best_weights=True,
)

# Define the directory containing the images
name = "Epidural"
Epidural_DIR = "CTbrain/merged_images/augminted/doubled/" + name
Features_path = name + "_features.npy"
Labels_path = name + "_labels.npy"
matrix_filename = "results/" + name + "_ANN.png"
model_name = "models/" + name + "_ANN.h5"

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

# Define ANN model
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN model
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[model_checkpoint, early_stopping],  # Add callbacks
)

# Evaluate the ANN model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy for ANN: {accuracy}')

# Save the trained CNN model
model.save(model_name)

# Create and save the loss graph
plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(os.path.join(graphs_dir, name + "_loss_graph.png"))
plt.show()

# Create and save the accuracy graph
plt.figure(figsize=(12, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(graphs_dir, name + "_accuracy_graph.png"))
plt.show()

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
