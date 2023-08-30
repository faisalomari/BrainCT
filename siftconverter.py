import os
import cv2
import numpy as np 

# Function to convert images to SIFT features and export
def convert_images_to_sift(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            input_path = os.path.join(input_dir, filename)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Create a SIFT object
            sift = cv2.SIFT_create()

            # Detect and compute SIFT features
            keypoints, descriptors = sift.detectAndCompute(image, None)

            # Save the SIFT descriptors to a file
            output_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.sift')
            np.savetxt(output_path, descriptors, delimiter=',', fmt='%f')

            print(f'Converted and saved SIFT features for: {input_path}')

# Input and output directories
input_images_dir = "CTbrain/merged_images/Epidural/without"  # Replace with the path to your input images directory
output_sift_dir = "CTbrain/merged_images/Epidural/without_sift"    # Replace with the path to your output SIFT directory

convert_images_to_sift(input_images_dir, output_sift_dir)
