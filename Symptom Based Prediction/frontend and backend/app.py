import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/predict',methods=['POST'])
def predict():
    BreathingProblem=request.form["brp"]
    Fever = request.form["fever"]
    DryCough = request.form["dry"]
    SoreThroat = request.form["sore"]
    RunningNose = request.form["rn"]
    Asthma = request.form["ast"]
    Headache = request.form["hde"]
    Abroad = request.form["abrd"]
    CWCP = request.form["contact"]
    ALG = request.form["largeg"]
    VLEP = request.form["visit"]
    FWPP = request.form["work"]
    x=np.array([[BreathingProblem,Fever,DryCough,SoreThroat,RunningNose,Asthma,Headache,Abroad,CWCP,ALG,VLEP,FWPP]])
    prediction = model.predict(x)
    if prediction==0:
        return render_template('base.html', prediction_text='NO COVID')
    elif prediction==1:
        return render_template('base.html', prediction_text='COVID')





if __name__ == "__main__":
    app.run(debug=True)