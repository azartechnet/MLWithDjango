import tensorflow as tf
import numpy as np
import requests

from PIL import Image
from io import BytesIO

# Step 1: External Image URLs

cat_urls = [
    "https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=300",
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300"
]

dog_urls = [
    "https://images.unsplash.com/photo-1552053831-71594a27632d?w=300",
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=300"
]

# Step 2: Load Images

X = []
y = []

def load_images(urls, label):

    for url in urls:

        response = requests.get(url)

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        image = image.resize((128, 128))

        X.append(np.array(image))
        y.append(label)


# Step 3: Assign Labels

# Cat = 0
# Dog = 1

load_images(cat_urls, 0)
load_images(dog_urls, 1)

X = np.array(X, dtype="float32")
y = np.array(y)

print("Images:", X.shape)
print("Labels:", y)

# Step 4: Create CNN Model

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128,128,3)),

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(
        32, (3,3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        64, activation="relu"
    ),

    tf.keras.layers.Dense(
        1, activation="sigmoid"
    )
])

# Step 5: Compile Model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Step 6: Train Model

model.fit(
    X,
    y,
    epochs=10
)

# Step 7: Save Model

model.save("cat_dog_model.keras")

print("Training Completed!")