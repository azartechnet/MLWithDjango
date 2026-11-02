
import numpy as np
from django.shortcuts import render
from sklearn.linear_model import LogisticRegression

# Training dataset
# Features: Study Hours, Attendance
X = np.array([
    [1, 40],
    [2, 50],
    [3, 60],
    [4, 70],
    [5, 80],
    [6, 90],
    [7, 100]
])

# Result: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 1])

# Training the model
model = LogisticRegression()
model.fit(X, y)


def home(request):

    result = None
    probability = None
    error = None

    if request.method == 'POST':
        try:
            study_hours = float(request.POST['study_hours'])
            attendance = float(request.POST['attendance'])

            input_data = np.array([[study_hours, attendance]])

            prediction = model.predict(input_data)

            probability = model.predict_proba(input_data)[0][1]

            result = 'Pass' if prediction[0] == 1 else 'Fail'

        except (ValueError, KeyError):
            error = 'Invalid input. Please enter valid numbers.'

    # This return must be outside the IF condition
    return render(request, 'home.html', {
        'result': result,
        'probability': probability,
        'error': error
    })
