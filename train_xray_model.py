import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set up the path to the data folder on D: drive
data_dir = 'D:/AI_Training/chest_xray_data/'
csv_file = os.path.join(data_dir, 'preprocessed_data.csv')

# Load the preprocessed CSV file containing image paths and labels
data = pd.read_csv(csv_file)

# Feature map for the TFRecord parsing
feature_map = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'No Finding': tf.io.FixedLenFeature([], tf.int64),
    'Atelectasis': tf.io.FixedLenFeature([], tf.int64),
    'Consolidation': tf.io.FixedLenFeature([], tf.int64),
    'Infiltration': tf.io.FixedLenFeature([], tf.int64),
    'Pneumothorax': tf.io.FixedLenFeature([], tf.int64),
    'Edema': tf.io.FixedLenFeature([], tf.int64),
    'Emphysema': tf.io.FixedLenFeature([], tf.int64),
    'Fibrosis': tf.io.FixedLenFeature([], tf.int64),
    'Effusion': tf.io.FixedLenFeature([], tf.int64),
    'Pneumonia': tf.io.FixedLenFeature([], tf.int64),
    'Pleural_Thickening': tf.io.FixedLenFeature([], tf.int64),
    'Cardiomegaly': tf.io.FixedLenFeature([], tf.int64),
    'Nodule': tf.io.FixedLenFeature([], tf.int64),
    'Mass': tf.io.FixedLenFeature([], tf.int64),
    'Hernia': tf.io.FixedLenFeature([], tf.int64)
}

# Function to parse TFRecord files
def parse_tfrecord_fn(example):
    return tf.io.parse_single_example(example, feature_map)

# Function to decode and process the image
def process_record(record):
    image = tf.image.decode_jpeg(record['image'], channels=1)
    image = tf.image.resize(image, [300, 300])  # Resize to smaller size to reduce memory usage
    image = image / 255.0  # Normalize pixel values
    label = [record['No Finding'], record['Atelectasis'], record['Consolidation'], record['Infiltration'],
             record['Pneumothorax'], record['Edema'], record['Emphysema'], record['Fibrosis'], record['Effusion'],
             record['Pneumonia'], record['Pleural_Thickening'], record['Cardiomegaly'], record['Nodule'],
             record['Mass'], record['Hernia']]
    return image, label

# List TFRecord files from 000-438 to 247-438 and from 248-437 to 255-437
tfrecord_files = []

# Add files ending with 438
tfrecord_files += [os.path.join(data_dir, f'{i:03d}-438.tfrec') for i in range(248) if os.path.exists(os.path.join(data_dir, f'{i:03d}-438.tfrec'))]

# Add files ending with 437
tfrecord_files += [os.path.join(data_dir, f'{i:03d}-437.tfrec') for i in range(248, 256) if os.path.exists(os.path.join(data_dir, f'{i:03d}-437.tfrec'))]

print(f"Number of TFRecord files found: {len(tfrecord_files)}")

# Create a dataset from the available TFRecord files
raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

# Map the dataset to the parser function and process it
dataset = raw_dataset.map(lambda x: process_record(parse_tfrecord_fn(x)))

# Split data into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(data))
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Batch and shuffle the datasets
BATCH_SIZE = 16  # Reduce batch size to prevent memory overload
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Model definition (a simple CNN model)
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)),  # Reduced image size
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(15, activation='sigmoid')  # Output layer for 15 classes
    ])
    return model

# Compile the model
model = create_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Multi-label classification problem
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Save the trained model to the D: drive
model_save_dir = os.path.join(data_dir, 'models/')
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
model.save(os.path.join(model_save_dir, 'xray_model.h5'))

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(os.path.join(model_save_dir, 'accuracy_plot.png'))
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(os.path.join(model_save_dir, 'loss_plot.png'))
plt.show()
