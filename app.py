from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler and model
scaler = pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        # Extract form data
        Pregnancies = int(request.form.get("Pregnancies", 0))
        Glucose = float(request.form.get('Glucose', 0))
        BloodPressure = float(request.form.get('BloodPressure', 0))
        SkinThickness = float(request.form.get('SkinThickness', 0))
        Insulin = float(request.form.get('Insulin', 0))
        BMI = float(request.form.get('BMI', 0))
        DiabetesPedigreeFunction = float(
            request.form.get('DiabetesPedigreeFunction', 0))
        Age = float(request.form.get('Age', 0))

        # Prepare data for prediction
        new_data = scaler.transform(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        predict = model.predict(new_data)

        # Determine result
        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'

        # Redirect to the result page with values
        return redirect(url_for('result',
                                result=result,
                                Pregnancies=Pregnancies,
                                Glucose=Glucose,
                                BloodPressure=BloodPressure,
                                SkinThickness=SkinThickness,
                                Insulin=Insulin,
                                BMI=BMI,
                                DiabetesPedigreeFunction=DiabetesPedigreeFunction,
                                Age=Age))

    except Exception as e:
        return f"Error occurred: {e}"


@app.route('/result')
def result():
    result = request.args.get('result')
    Pregnancies = request.args.get('Pregnancies')
    Glucose = request.args.get('Glucose')
    BloodPressure = request.args.get('BloodPressure')
    SkinThickness = request.args.get('SkinThickness')
    Insulin = request.args.get('Insulin')
    BMI = request.args.get('BMI')
    DiabetesPedigreeFunction = request.args.get('DiabetesPedigreeFunction')
    Age = request.args.get('Age')

    return render_template('result.html',
                           result=result,
                           Pregnancies=Pregnancies,
                           Glucose=Glucose,
                           BloodPressure=BloodPressure,
                           SkinThickness=SkinThickness,
                           Insulin=Insulin,
                           BMI=BMI,
                           DiabetesPedigreeFunction=DiabetesPedigreeFunction,
                           Age=Age)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
