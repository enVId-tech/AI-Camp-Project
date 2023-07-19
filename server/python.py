from flask import Flask, render_template, request, redirect, url_for, session, send_file
import requests
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def main():
    return render_template("index.html")

API_URL = "https://api-inference.huggingface.co/models/macharya/predict_leaf_disease"
headers = {"Authorization": "Bearer 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"}

app.config["IMAGE_UPLOADS"] = "./"
app.config['UPLOAD_FOLDER'] = 'static/images/'

def query(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

@app.route('/predict', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        output = query(image_path)  # Pass image_path to the query function
        return render_template("result.html", name=output, img=f.filename)
    
PORT = 3000
app.run(host='0.0.0.0', port=PORT)