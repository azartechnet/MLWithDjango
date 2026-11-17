import numpy as np
from tensorflow import keras

# Load trained model
model = keras.models.load_model("fruit_model.keras")

# Load test image
image = keras.utils.load_img(
    # "banana1.jpg",
    "apple1.jpg",
    target_size=(128, 128)
)

# Convert image
image_array = keras.utils.img_to_array(image)

image_array = np.expand_dims(image_array, axis=0)

# Predict
prediction = model.predict(image_array)

print("Prediction Value:", prediction[0][0])

if prediction[0][0] >= 0.5:
    print("Banana Detected")
else:
    print("Apple Detected")