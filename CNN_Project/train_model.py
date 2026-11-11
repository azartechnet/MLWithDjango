
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Step 1: Load training dataset
train_data = keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(128, 128),
    batch_size=32,
    label_mode="binary"
)

# Step 2: Load validation dataset
val_data = keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=(128, 128),
    batch_size=32,
    label_mode="binary"
)

# Step 3: Create CNN model
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Normalize images
    layers.Rescaling(1./255),

    # Convolution layer
    layers.Conv2D(32, (3, 3), activation='relu'),

    # Pooling layer
    layers.MaxPooling2D((2, 2)),

    # Second convolution layer
    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.MaxPooling2D((2, 2)),

    # Flatten layer
    layers.Flatten(),

    # Dense layer
    layers.Dense(64, activation='relu'),

    # Output layer
    layers.Dense(1, activation='sigmoid')
])

# Step 4: Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Step 6: Save model
model.save("cat_dog_model.keras")

print("CNN Model Training Completed")
