import numpy as np
from tensorflow import keras

model = keras.models.load_model("cat_dog_model.keras")

image = keras.utils.load_img(
    "test.jpg",
    target_size=(128, 128)
)

image_array = keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

prediction = model.predict(image_array)

if prediction[0][0] >= 0.5:
    print("Dog Detected")
else:
    print("Cat Detected")