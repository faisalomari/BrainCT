import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
import numpy as np
import csv
from tensorflow.keras.models import load_model
from skimage.feature import hog
from skimage import io
import tensorflow as tf
# Function to extract HOG features from an image
def extract_hog(image_path):
    image = io.imread(image_path, as_gray=True)
    fd = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')
    fd = fd.reshape(1, -1)  # Reshape for model input
    return fd
# Function to load models and make predictions
def predict_image():
    image_path = image_path_entry.get()
    device_choice = device_var.get()
    
    # Validate the image path
    if not image_path or not os.path.exists(image_path):
        result_label.config(text="Invalid image path", fg="red")
        return
    
    # Validate user input
    if device_choice not in ["GPU", "CPU"]:
        result_label.config(text="Invalid choice. Please choose either GPU or CPU.", fg="red")
        return
    
    # Disable GPU devices if CPU is chosen
    if device_choice == "CPU":
        tf.config.set_visible_devices([], 'GPU')
    
    # Extract HOG features from the selected image
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
    
    # Display predictions
    result_label.config(text="Predictions:")
    predictions_text = "\n".join([f"{disease}: {prediction}" for disease, prediction in predictions.items()])
    predictions_label.config(text=predictions_text)
    
    # Save the predictions to a CSV file with diseases in the first row and predictions in the second row
    output_file = "predictions.csv"
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write disease names in the first row
        writer.writerow(list(predictions.keys()))
        
        # Write predictions in the second row
        writer.writerow(list(predictions.values()))
    
    result_label.config(text=f"Predictions saved to {output_file}", fg="green")

# Function to open a file dialog for image selection
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        image_path_entry.delete(0, tk.END)
        image_path_entry.insert(0, file_path)

# Create the main window
root = tk.Tk()
root.title("Medical Image Predictor")

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

# Create and configure GUI elements
image_path_label = tk.Label(root, text="Select an image:")
image_path_entry = tk.Entry(root, width=40)
browse_button = tk.Button(root, text="Browse", command=browse_image)
device_label = tk.Label(root, text="Choose device (GPU/CPU):")
device_var = tk.StringVar(value="CPU")
device_combobox = ttk.Combobox(root, textvariable=device_var, values=["CPU", "GPU"])
predict_button = tk.Button(root, text="Predict", command=predict_image)
result_label = tk.Label(root, text="")
predictions_label = tk.Label(root, text="")

# Arrange GUI elements in the window
image_path_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
image_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
browse_button.grid(row=0, column=2, padx=5, pady=5)
device_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
device_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
predict_button.grid(row=2, column=1, padx=5, pady=10)
result_label.grid(row=3, column=0, columnspan=3, padx=10, pady=5)
predictions_label.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

# Start the GUI main loop
root.mainloop()
