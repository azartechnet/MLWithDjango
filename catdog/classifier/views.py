
import tensorflow as tf
import numpy as np

from PIL import Image
from io import BytesIO
from pathlib import Path

from django.shortcuts import render

# Load trained CNN model
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "cat_dog_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def home(request):

    context = {}

    if request.method == "POST":

        uploaded_image = request.FILES.get("image")

        if uploaded_image:

            try:
                # Read uploaded image
                image = Image.open(
                    BytesIO(uploaded_image.read())
                ).convert("RGB")

                # Resize image
                image = image.resize((128, 128))

                # Convert into NumPy array
                image_array = np.array(
                    image, dtype="float32"
                )

                # Add batch dimension
                image_array = np.expand_dims(
                    image_array, axis=0
                )

                # CNN Prediction
                prediction = model.predict(
                    image_array, verbose=0
                )[0][0]

                # Classification
                if prediction >= 0.5:
                    result = "Dog"
                    confidence = prediction * 100
                else:
                    result = "Cat"
                    confidence = (1 - prediction) * 100

                context = {
                    "result": result,
                    "confidence": round(
                        float(confidence), 2
                    ),
                    "uploaded_image": uploaded_image,
                }

            except Exception as e:

                context["error"] = (
                    "Unable to process image: " + str(e)
                )

        else:
            context["error"] = "Please upload an image."

    return render(request, "index.html", context)
