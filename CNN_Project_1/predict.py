import numpy as np
from tensorflow import keras

# Step 1: Load Model

model = keras.models.load_model(
    "cat_dog_model.keras"
)

# Step 2: Load Test Image

image = keras.utils.load_img(
    "test1.jpg",
    target_size=(128,128)
)

# Step 3: Convert Image

image_array = keras.utils.img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)

# Step 4: Predict

prediction = model.predict(image_array)

# Step 5: Display Result

if prediction[0][0] >= 0.5:

    print("Dog Detected")

else:

    print("Cat Detected")