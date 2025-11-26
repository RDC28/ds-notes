from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        import pickle
        import numpy as np

        model_filename = '../../models/E_scooter/E_scooter_lr_model.pkl'
        scaler_filename = '../../models/E_scooter/E_scooter_scaler.pkl'

        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        
        with open(scaler_filename, 'rb') as file:
            loaded_scaler = pickle.load(file)

        inp = np.array([[
               float(request.form.get('RideDistance_km')),
               float(request.form.get('AvgSpeed_kmh')),
               float(request.form.get('Temperature_C')),
               float(request.form.get('RiderWeight_kg'))]])
        print("Model loaded successfully.")
        res = loaded_model.predict(loaded_scaler.transform(inp))
        print(loaded_model)
        print(res)
        return render_template('result.html', result = res)
            
    else:
        return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
        app.run(debug=True)