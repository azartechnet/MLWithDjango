from django.shortcuts import render
from sklearn.linear_model import LinearRegression
import numpy as np

#Training dataset:hours studied
X=np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])
#Corresponding student marks
y=np.array([15,25,35,45,55,65,75,85])

#create and train model
model=LinearRegression()
model.fit(X,y)

def home(request):
    prediction=None
    error=None
    if request.method=="POST":
        try:
            hours=float(request.POST.get("hours",""))
            if not np.isfinite(hours) or not 0<=hours <=10:
                raise ValueError
            # prediction=round(float(model.predict([hours]))[0],2)
            prediction = round(float(model.predict([[hours]])[0]), 2)
        except (ValueError,TypeError):
            error="Enter valid study hours between 0 and 10"
    return render(request,"prediction/home.html",{"prediction":prediction,"error":error})
