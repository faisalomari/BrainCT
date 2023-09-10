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
import joblib
import copy

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
    selected_model = model_var.get()
    
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
    
    if "All" in selected_model:
        for disease, model_path in model_paths.items():
            p = 0
            for model in models:
                if(model == "LSTM"):
                    feature2 = copy.deepcopy(feature)
                    feature2 = feature2.reshape(feature.shape[0], 1, feature.shape[1])
                    path = model_path + model + ".h5"
                    with tf.device(f"/{device_choice}:0"):
                        curr_model = load_model(path)
                        prediction = curr_model.predict(feature2)
                        p += prediction[0][0]
                        del curr_model
                        tf.keras.backend.clear_session()
                elif(model == "Logistic Regression"):
                    path = model_path + "LR.h5"
                    with tf.device(f"/{device_choice}:0"):
                        curr_model = joblib.load(path)
                        prediction = curr_model.predict(feature)
                        del curr_model
                        tf.keras.backend.clear_session()
                elif(model == "SVM"):
                    path = model_path + model + ".h5"
                    with tf.device(f"/{device_choice}:0"):
                        curr_model = joblib.load(path)
                        prediction = curr_model.predict(feature)
                        p = p + (1 if prediction[0]==1 else 0)
                        del curr_model
                        tf.keras.backend.clear_session()
                else:
                    path = model_path + model + ".h5"
                    with tf.device(f"/{device_choice}:0"):
                        curr_model = load_model(path)
                        prediction = curr_model.predict(feature)
                        p += prediction[0][0]
                        del curr_model
                        tf.keras.backend.clear_session()
            
            binary_prediction = 1 if p > len(models)/2 else 0
            predictions[disease] = binary_prediction
    else:
        if(selected_model == "LSTM"):
            feature = feature.reshape(feature.shape[0], 1, feature.shape[1])
        for disease, model_path in model_paths.items():
            if(selected_model == "Logistic Regression"):
                path = model_path + "LR.h5"
                with tf.device(f"/{device_choice}:0"):
                    curr_model = joblib.load(path)
                    prediction = curr_model.predict(feature)
                    predictions[disease] = prediction[0]
                    del curr_model
                    tf.keras.backend.clear_session()
            elif(selected_model == "SVM"):
                path = model_path + selected_model + ".h5"
                with tf.device(f"/{device_choice}:0"):
                    curr_model = joblib.load(path)
                    prediction = curr_model.predict(feature)
                    predictions[disease] = prediction[0]
                    del curr_model
                    tf.keras.backend.clear_session()

            else:
                path = model_path + selected_model + ".h5"
                with tf.device(f"/{device_choice}:0"):
                    curr_model = load_model(path)
                    prediction = curr_model.predict(feature)
                    binary_prediction = 1 if prediction[0][0] > 0.5 else 0
                    predictions[disease] = binary_prediction
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
root.title("Medical Image Identifier")

# Set the scale factor for size increase
scale_factor = 2

# Define the paths to the H5 models
model_paths = {
    "Epidural": "models/Epidural_",
    "Fracture_Yes_No": "models/Fracture_Yes_No_",
    "Intraparenchymal": "models/Intraparenchymal_",
    "Intraventricular": "models/Intraventricular_",
    "No_Hemorrhage": "models/No_Hemorrhage_",
    "Subarachnoid": "models/Subarachnoid_",
    "Subdural": "models/Subdural_",
}

models = {
    "CNN",
    "ANN",
    "LSTM",
    "SVM",
    "Logistic Regression"
}

# Create and configure GUI elements with increased size
image_path_label = tk.Label(root, text="Select an image:", font=("Arial", 12*scale_factor))
image_path_entry = tk.Entry(root, width=20*scale_factor, font=("Arial", 12*scale_factor))
browse_button = tk.Button(root, text="Browse", command=browse_image, font=("Arial", 12*scale_factor))
device_label = tk.Label(root, text="Choose device (GPU/CPU):", font=("Arial", 12*scale_factor))
device_var = tk.StringVar(value="CPU")
device_combobox = ttk.Combobox(root, textvariable=device_var, values=["CPU", "GPU"], font=("Arial", 12*scale_factor))

all_models = list(models)
all_models.insert(0, "All")
model_label = tk.Label(root, text="Select a model:", font=("Arial", 12*scale_factor))
model_var = tk.StringVar()
model_var.set("All")  # Default to "All" models
model_combobox = ttk.Combobox(root, textvariable=model_var, values=all_models, state="readonly", font=("Arial", 12*scale_factor))

predict_button = tk.Button(root, text="Predict", command=predict_image, font=("Arial", 12*scale_factor))
result_label = tk.Label(root, text="", font=("Arial", 12*scale_factor))
predictions_label = tk.Label(root, text="", font=("Arial", 12*scale_factor))

# Arrange GUI elements in the window
image_path_label.grid(row=0, column=0, padx=10*scale_factor, pady=5*scale_factor, sticky="e")
image_path_entry.grid(row=0, column=1, padx=5*scale_factor, pady=5*scale_factor, sticky="w")
browse_button.grid(row=0, column=2, padx=5*scale_factor, pady=5*scale_factor)
device_label.grid(row=1, column=0, padx=10*scale_factor, pady=5*scale_factor, sticky="e")
device_combobox.grid(row=1, column=1, padx=5*scale_factor, pady=5*scale_factor, sticky="w")
model_label.grid(row=2, column=0, padx=10*scale_factor, pady=5*scale_factor, sticky="e")
model_combobox.grid(row=2, column=1, padx=5*scale_factor, pady=5*scale_factor, sticky="w")
predict_button.grid(row=3, column=1, padx=5*scale_factor, pady=10*scale_factor)
result_label.grid(row=4, column=0, columnspan=3, padx=10*scale_factor, pady=5*scale_factor)
predictions_label.grid(row=5, column=0, columnspan=3, padx=10*scale_factor, pady=5*scale_factor)

# Set the size of the main window
root.geometry(f"{800*scale_factor}x{600*scale_factor}")

# Start the GUI main loop
root.mainloop()
