import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# Define constants
IMAGE_SIZE = (650, 1300)
BATCH_SIZE = 32

# Data and output directories
data_dir = "CTbrain/merged_images/Subdural"  # Update with your data directory
output_dir = "CTbrain/merged_images/augminted/Subdural"  # Update with your desired output directory


num_files1 = sum(1 for _ in os.listdir(data_dir+"/without") if os.path.isfile(os.path.join(data_dir+"/without", _)))
print("Number of files without:", num_files1)
num_files2 = sum(1 for _ in os.listdir(data_dir+"/with") if os.path.isfile(os.path.join(data_dir+"/with", _)))
print("Number of files with:", num_files2)

times = int(num_files1/num_files2)
print("TIMES:", times)

# Create an ImageDataGenerator for augmentation
augmentation_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# List of subdirectories in data_dir
subdirectories = ['with']

for subdirectory in subdirectories:
    label_dir = os.path.join(data_dir, subdirectory)
    label_output_dir = os.path.join(output_dir, subdirectory)

    if not os.path.exists(label_output_dir):
        os.makedirs(label_output_dir)

    # Get list of image file paths
    label_paths = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir) if filename.endswith('.jpg')]

    for i, image_path in enumerate(label_paths):
        image = load_img(image_path, target_size=IMAGE_SIZE)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        # Generate augmented images
        for j in range(times):
            augmented_image = augmentation_generator.random_transform(image_array[0])
            augmented_image = array_to_img(augmented_image)

            # Save augmented image
            output_path = os.path.join(label_output_dir, f'{i}_aug{j}.jpg')
            augmented_image.save(output_path)

            print(f'Saved augmented image: {output_path}')
