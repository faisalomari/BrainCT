import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from numba import cuda
# cuda.select_device(0)
# cuda.close()
# Define constants
IMAGE_SIZE = (650, 650)
BATCH_SIZE = 6
EPOCHS = 5
LEARNING_RATE = 0.001
DATASET_DIR = "CTbrain/data_splitted_brain/Epidural"  # Update this to your dataset path

# Get list of image file paths
with_images = [os.path.join(DATASET_DIR, "with", filename) for filename in os.listdir(os.path.join(DATASET_DIR, "with"))]
without_images = [os.path.join(DATASET_DIR, "without", filename) for filename in os.listdir(os.path.join(DATASET_DIR, "without"))]

# Create train, validation, and test data splits
with_train, with_test = train_test_split(with_images, test_size=0.1, random_state=42)
without_train, without_test = train_test_split(without_images, test_size=0.1, random_state=42)

# Further split validation from remaining training data
with_train, with_val = train_test_split(with_train, test_size=0.25, random_state=42)
without_train, without_val = train_test_split(without_train, test_size=0.25, random_state=42)

# Combine train and validation splits
train_images = with_train + without_train
val_images = with_val + without_val
test_images = with_test + without_test

# Create a DataFrame with image paths and labels
train_df = pd.DataFrame({"image_path": train_images, "label": [1] * len(with_train) + [0] * len(without_train)})
val_df = pd.DataFrame({"image_path": val_images, "label": [1] * len(with_val) + [0] * len(without_val)})
test_df = pd.DataFrame({"image_path": test_images, "label": [1] * len(with_test) + [0] * len(without_test)})

# Count the number of images for each label in train, val, and test sets
train_counts = train_df["label"].value_counts()
val_counts = val_df["label"].value_counts()
test_counts = test_df["label"].value_counts()

# Create a DataFrame to display the counts in a table
counts_table = pd.DataFrame({
    "Label": train_counts.index,
    "Train": train_counts.values,
    "Validation": val_counts.values,
    "Test": test_counts.values
})

# Display the table
print(counts_table)

# Create data generators using flow_from_dataframe
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="image_path",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

# Create InceptionV3 base model
base_model = InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# Evaluate the model on the test set
test_results = model.evaluate(test_generator)
print("Test loss:", test_results[0])
print("Test accuracy:", test_results[1])

# Get predictions for the test set
test_predictions = model.predict(test_generator)

# Convert predictions to binary labels (0 or 1)
test_predicted_labels = (test_predictions > 0.5).astype(int)

# Get unique labels from the test set
unique_labels = test_df["label"].unique()
print(unique_labels)

# Calculate the number of correct predictions for each label
correct_predictions = {}
total_samples_per_label = {}

for label in unique_labels:
    label_name = str(label)  # Convert label to string if not already
    label_mask = test_df["label"] == label
    correct_predictions[label_name] = (test_predicted_labels[label_mask] == label).sum()
    total_samples_per_label[label_name] = label_mask.sum()

# Calculate the percentage of correct predictions for each label
percentage_correct_per_label = {}

for label, correct_count in correct_predictions.items():
    total_samples = total_samples_per_label[label]
    percentage_correct_per_label[label] = (correct_count / total_samples) * 100


print(percentage_correct_per_label)
# Create a DataFrame to display the results in a table
accuracy_table = pd.DataFrame({
    "Label": percentage_correct_per_label.keys(),
    "Total Samples": total_samples_per_label.values(),
    "Correct Predictions": correct_predictions.values(),
    "Accuracy (%)": percentage_correct_per_label.values()
})

# Display the accuracy table
print(accuracy_table)
