import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# External image URLs
apple_urls = [
    "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=300",
    "https://images.unsplash.com/photo-1570913149827-d2ac84ab3f9a?w=300"
]

banana_urls = [
    "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=300",
    "https://images.unsplash.com/photo-1528825871115-3581a5387919?w=300"
]

X = []
y = []

# Download and process images
def load_images(urls, label):

    for url in urls:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        image = image.resize((128, 128))

        X.append(np.array(image))
        y.append(label)


load_images(apple_urls, 0)
load_images(banana_urls, 1)

# Convert into NumPy arrays
X = np.array(X, dtype="float32")
y = np.array(y)

# CNN Model
model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(X, y, epochs=10)

# Save model
model.save("fruit_model.keras")

print("Fruit Model Training Completed!")