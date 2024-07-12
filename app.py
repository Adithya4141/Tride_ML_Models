from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
with open('models/DEC_regressor.pkl', 'rb') as f:
    DEC_regressor = pickle.load(f)

with open('models/Engine_health_svc_classifier.pkl', 'rb') as f:
    Engine_health_svc_classifier = pickle.load(f)

with open('models/intrsuion_detection.pkl', 'rb') as f:
    intrsuion_detection = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Engine_health_condition', methods=['GET', 'POST'])
def Engine_health_condition():
    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        a = Engine_health_svc_classifier.predict(final_features)
        if a[0] == 0:
            prediction = "Engine needs maintenance"
        else:
            prediction = "Engine is in Normal Condition"
        return render_template('Engine_health_condition.html', prediction=prediction)
    return render_template('Engine_health_condition.html')

@app.route('/Intrusion_Detection', methods=['GET', 'POST'])
def Intrusion_Detection():
    if request.method == 'POST':
        features = [request.form[f'feature{i}'] for i in range(4)]
        prediction = intrsuion_detection.predict(np.array([features]))
        return render_template('Intrusion_Detection.html', prediction=prediction[0])
    return render_template('Intrusion_Detection.html')

@app.route('/Dynamic_Energy_consumption', methods=['GET', 'POST'])
def Dynamic_Energy_consumption():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        a = DEC_regressor.predict(final_features)
        prediction = f"{a[0]}% of SOC is consumed"
        return render_template('Dynamic_Energy_consumption.html', prediction=prediction)
    return render_template('Dynamic_Energy_consumption.html')


if __name__ == "__main__":
    app.run(debug=True)
