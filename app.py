import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('gb_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    Brand = request.form.get('Brand')    
    Processor_Brand = request.form.get('Processor_Brand')
    Processor_Type=request.form.get('Processor_Type')
    Storage_SSD =request.form.get('Storage_SSD')
    Storage_HDD =request.form.get('Storage_HDD')
    RAM_GB=request.form.get('RAM_GB')
    RAM_Type=request.form.get('RAM_Type')
    OS =request.form.get('OS')
    Display_Size =request.form.get('Display_Size')
    Display_Type =request.form.get('Display_Type')
    Office =request.form.get('Office')
    Warranty = request.form.get('Warranty')

    inputs = pd.DataFrame([[Brand,
                            Processor_Brand,
                            Processor_Type, 
                            Storage_SSD,
                            Storage_HDD, 
                            RAM_GB, 
                            RAM_Type,
                            OS, 
                            Display_Size,
                            Display_Type,
                            Office,
                            Warranty]], 
                columns=['Brand',
                         'Processor_Brand', 
                         'Processor_Type',
                         'Storage_SSD',
                         'Storage_HDD',
                         'RAM_GB',
                         'RAM_Type',
                         'OS', 
                         'Display_Size',
                         'Display_Type', 
                         'Office',
                         'Warranty'])
    

    prediction = model.predict(inputs)

    output = round(prediction[0], 2)

    return render_template('index1.html', prediction_text='Price of Laptop is :  ₹ {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)