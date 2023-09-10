import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from skimage.feature import hog
from skimage import io
import tensorflow as tf
import csv

#choose of device (GPU or CPU)
device = "CPU"
device_choice = device.strip().lower()

# Validate user input
if device_choice not in ["gpu", "cpu"]:
    print("Invalid choice. Please choose either GPU or CPU.")
    exit()

# Disable GPU devices if CPU is chosen
if device_choice == "cpu":
    tf.config.set_visible_devices([], 'GPU')

# Define the paths to the H5 models
model_paths = {
    "Epidural": "models/Epidural_CNN.h5",
    "Fracture_Yes_No": "models/Fracture_Yes_No_CNN.h5",
    "Intraparenchymal": "models/Intraparenchymal_CNN.h5",
    "Intraventricular": "models/Intraventricular_CNN.h5",
    "No_Hemorrhage": "models/No_Hemorrhage_CNN.h5",
    "Subarachnoid": "models/Subarachnoid_CNN.h5",
    "Subdural": "models/Subdural_CNN.h5",
}

def extract_hog(image_path):
    image = io.imread(image_path, as_gray=True)
    fd = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')
    fd = fd.reshape(1, -1)  # Reshape for model input
    return fd

image_path = "CTbrain/merged_images/Epidural/with/0.jpg"  # The path to your image
feature = extract_hog(image_path)
predictions = {}

for disease, model_path in model_paths.items():
    with tf.device(f"/{device_choice}:0"):  # Ensure the code runs on the chosen device (GPU/CPU)
        curr_model = load_model(model_path)
        prediction = curr_model.predict(feature)
        binary_prediction = 1 if prediction[0][0] > 0.5 else 0
        predictions[disease] = binary_prediction

        # Unload the model and clear GPU memory
        del curr_model
        tf.keras.backend.clear_session()

for disease, prediction in predictions.items():
    print(f"{disease}: {prediction}")

# Save the predictions to a CSV file with diseases in the first row and predictions in the second row
output_file = "predictions.csv"
with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write disease names in the first row
    writer.writerow(list(predictions.keys()))
    
    # Write predictions in the second row
    writer.writerow(list(predictions.values()))

print(f"Predictions saved to {output_file}")