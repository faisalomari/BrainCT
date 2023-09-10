import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd 
import os

def count_elements_in_folder(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError("Folder does not exist")

    elements = os.listdir(folder_path)
    num_elements = len(elements)
    return num_elements

def combine_images(image_path_left, image_path_right, destination_path):
    # Load the two images
    image_left = Image.open(image_path_left).convert('L')  # Convert to grayscale
    image_right = Image.open(image_path_right).convert('L')  # Convert to grayscale

    # Convert images to NumPy arrays
    array_left = np.array(image_left)
    array_right = np.array(image_right)

    # Check if the images have the same height
    if array_left.shape[0] != array_right.shape[0]:
        raise ValueError("Images must have the same height")

    # Create a new empty image for the combined result
    combined_width = array_left.shape[1] + array_right.shape[1]
    combined_image = np.zeros((array_left.shape[0], combined_width), dtype=np.uint8)

    # Fill in the combined image with the pixel values from the two images
    combined_image[:, :array_left.shape[1]] = array_left
    combined_image[:, array_left.shape[1]:] = array_right

    # Create a PIL image from the combined NumPy array
    final_image = Image.fromarray(combined_image)

    # Save the final image to the destination path
    final_image.save(destination_path)

currentDir = Path('/home/faisal/Desktop/BrainDL/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0')
datasetDir = str(Path(currentDir, 'Patients_CT'))

# Reading labels
hemorrhage_diagnosis_df = pd.read_csv(
    Path(currentDir, 'hemorrhage_diagnosis.csv')
)
hemorrhage_diagnosis_array = hemorrhage_diagnosis_df.values  # Use .values instead of ._get_values

t1 = Path('/home/faisal/Desktop/BrainDL/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/data_splitted_bone')
t2 = Path('/home/faisal/Desktop/BrainDL/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/data_splitted_brain')
subdirectories = [d for d in os.listdir(t1) if os.path.isdir(os.path.join(t1, d))]
print(subdirectories)

outDir = Path('/home/faisal/Desktop/BrainDL/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/merged_images')

for sub in subdirectories:
    dir1_with = os.path.join(str(t1), sub, 'with')
    dir2_with = os.path.join(str(t2), sub, 'with')
    out_with = os.path.join(str(outDir), sub, 'with')
    dir1_without = os.path.join(str(t1), sub, 'without')
    dir2_without = os.path.join(str(t2), sub, 'without')
    out_without = os.path.join(str(outDir), sub, 'without')
    # Get list of files in the 'with' subdirectories
    files_with = os.listdir(dir1_with)

    # Iterate over files
    for file_name in files_with:
        path1 = os.path.join(dir1_with, file_name)
        path2 = os.path.join(dir2_with, file_name)
        extension = '/' + file_name
        combine_images(path1, path2, out_with + extension)

    # Get list of files in the 'without' subdirectories
    files_without = os.listdir(dir1_without)

    # Iterate over files
    for file_name in files_without:
        path1 = os.path.join(dir1_without, file_name)
        path2 = os.path.join(dir2_without, file_name)
        extension = '/' + file_name
        combine_images(path1, path2, out_without + extension)