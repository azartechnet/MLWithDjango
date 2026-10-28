
from django.shortcuts import render
from sklearn.linear_model import LinearRegression
import numpy as np

# Training dataset
# House area in square feet
X = np.array([
    [500],
    [800],
    [1000],
    [1200],
    [1500],
    [1800],
    [2000]
])

# House prices in lakhs of rupees
y = np.array([15, 24, 30, 36, 45, 54, 60])

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)


def home(request):
    predicted_price = None
    area = ""

    if request.method == "POST":
        area = request.POST.get("area", "").strip()

        try:
            house_area = float(area)

            if not np.isfinite(house_area) or house_area <= 0:
                raise ValueError

            prediction = model.predict([[house_area]])[0]
            predicted_price = round(float(prediction), 2)

        except (ValueError, TypeError):
            predicted_price = None
            area = ""

    return render(request, "predictor/index.html", {
        "predicted_price": predicted_price,
        "area": area
    })
