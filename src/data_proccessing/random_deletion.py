# Import the os and random modules
import os
import random

# Define the folder path as a variable
folder_path = "CTbrain/merged_images/augminted/doubled/No_Hemorrhage2/with"

# Get the list of all files in the folder
files = os.listdir(folder_path)

# Filter the list to keep only the files with image extensions
# You can add or remove other image extensions as needed
image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
images = [f for f in files if f.endswith(tuple(image_extensions))]
print(len(images))
# Get the number of images to delete
# This will be half of the total number of images
num_delete = len(images) // 5

# Randomly select num_delete images from the list
images_to_delete = random.sample(images, num_delete)

# Loop through the selected images and delete them from the folder
for image in images_to_delete:
    # Get the full path of the image file
    image_path = os.path.join(folder_path, image)
    # Delete the image file
    os.remove(image_path)
    # Print a message to confirm the deletion
    print(f"Deleted {image} from {folder_path}")

# Print a message to indicate the end of the process
print(f"Successfully deleted {num_delete} images from {folder_path}")
