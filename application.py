from flask import Flask, request, app, render_template
from flask import Response
import pickle
import pandas as pd
import numpy as np

application = Flask(__name__)
app = application

scaler = pickle.load(open("Model/StandardScaler.pkl", "rb"))
model = pickle.load(open("Model/ModelForPrediction.pkl", "rb"))


## Route for HomePage
@app.route("/")
def index():
    return render_template("index.html")

## Route for single data point prediction
@app.route("/predata", methods = ["POST", "GET"])
def predict():
    result = ""

    if request.method == "POST":
        Pregency = int(request.form.get("Pregency"))
        Glucose = float(request.form.get("Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThikness = float(request.form.get("SkinThikness"))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = int(request.form.get("Age"))

        new_data = scaler.transform([[Pregency, Glucose, BloodPressure, SkinThikness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        predict = model.predict(new_data)

        if predict[0] == 1:
            result = "Diabatics"
        else:
            result = "NON Diabaics"

        return render_template("result.html", result = result)

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")