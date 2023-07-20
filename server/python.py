import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_cors import CORS
import requests
import uuid
import base64
import torch
from transformers import AutoModel, AutoTokenizer
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
# Directory where uploaded files will be stored
app.config["IMAGE_UPLOADS"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {
    "png", "jpg", "jpeg", "gif"
}  # Allowed file extensions

# API Configuration
API_URL = "https://huggingface.co/scoobydoo1688/LanguageIdentificationAICamp/blob/main/resnet50.bin"
# Replace with your actual API key
API_KEY = "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"
headers = {"Authorization": f"Bearer {API_KEY}"}
#######################################################
MODEL_NAME = "scoobydoo1688/LanguageIdentificationAICamp"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # Set the model to evaluation mode

def query_language_identification(image_data):
    """Send the image data to the language identification model and get the response."""
    files = {"image_data": image_data}

    # Perform inference using the pre-trained model
    inputs = tokenizer.decode(image_data)
    encoded_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Process the model output and return the language identification result
    # Replace this part with the logic specific to your model's output format
    # For example, if it's a classification task, you might use softmax and argmax
    # to get the predicted class label.
    language_id = torch.argmax(outputs.logits).item()
    language = "English" if language_id == 0 else "Non-English"

    return {"language": language}
#######################################################

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def generate_unique_filename(filename):
    """Generate a unique filename to avoid overwriting existing files."""
    unique_id = str(uuid.uuid4())[:8]
    return f"{secure_filename(filename)}_{unique_id}"


def query_language_identification(image_data):
    """Send the image data to the language identification API and get the response."""
    files = {"image_data": image_data}

    print(API_URL, headers, files)
    response = requests.post(API_URL, headers=headers, files=files)

    print(response)
    if response.status_code == 200:
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            return {"error": "Invalid JSON response from the API"}

    # Handle non-200 status codes
    return {"error": "Failed to get a valid response from the API"}


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def success():
    if request.method == 'POST':
        # Check if the request contains JSON data
        if request.is_json:
            request_data = request.get_json()
        else:
            return jsonify({"error": "Invalid request data. Expecting JSON."}), 400

        # Extract the image data from the JSON payload
        image_data = base64.b64decode(request_data.get("image_data", ""))

        # Check if the file is allowed based on its extension
        if not allowed_file(request_data.get("filename", "")):
            return jsonify({"error": "Invalid file type. Allowed extensions are: png, jpg, jpeg, gif."}), 400

        # Generate a unique filename to avoid overwriting existing files
        filename = generate_unique_filename(request_data.get("filename", ""))

        # Save the file to the designated upload folder
        with open(os.path.join(app.config["IMAGE_UPLOADS"], filename), "wb") as f:
            f.write(image_data)

        # Call the language identification API and get the response
        output = query_language_identification(image_data)

        return jsonify(output)


@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    """Safely serve the uploaded file from the 'uploads' directory."""
    return send_from_directory(app.config["IMAGE_UPLOADS"], filename)


if __name__ == "__main__":
    # Ensure the 'uploads' directory exists
    os.makedirs(app.config["IMAGE_UPLOADS"], exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
