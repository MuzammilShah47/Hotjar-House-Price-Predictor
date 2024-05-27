from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask('__name__')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    hotjar_site_id = os.getenv('HOTJAR_SITE_ID')
    return render_template('index.html', hotjar_site_id=hotjar_site_id)

@app.route('/predict', methods=["POST"])
def predict():
    hotjar_site_id = os.getenv('HOTJAR_SITE_ID')
    feature = [int(x) for x in request.form.values()]
    feature_final = np.array(feature).reshape(-1, 1)
    prediction = model.predict(feature_final)
    return render_template('index.html', hotjar_site_id=hotjar_site_id, prediction_text='Price of House will be Rs. {}'.format(int(prediction)))

if __name__ == '__main__':
    app.run(debug=True)

