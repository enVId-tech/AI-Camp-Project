import os
import io
from flask import Flask, render_template, request, send_from_directory, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50
from torchvision import models
from PIL import Image
from flask_cors import CORS
import requests
import uuid
import base64
from werkzeug.utils import secure_filename
import gradio as gr
from PIL import Image

app = Flask(__name__)
CORS(app)

# Configuration
# Directory where uploaded files will be stored
app.config["IMAGE_UPLOADS"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {
    "png", "jpg", "jpeg", "gif"}  # Allowed file extensions

# Model Configuration
MODEL_PATH = "server/model/resnet50.pth"

# Define the model class (you should replace this with your actual model class)


# Define the model class using ResNet50 as the base model
# class MyDataset(nn.Module):
#     def __init__(self):
#         super(MyDataset, self).__init__()
#         # Create a new instance of the ResNet50 model
#         self.resnet = resnet50(pretrained=False)  # Set pretrained=False to load only the architecture
# 
#         # Remove the last fully connected layer of ResNet50
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
# 
#     def forward(self, x):
#         # Perform forward pass through ResNet50
#         return self.resnet(x)

# Create an instance of your model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 5)

model.load_state_dict(torch.load('model\\resnet50.bin', map_location=torch.device('cpu')))

model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_function(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_path)

    input_img = image_transform(image)
    input_img_data = input_img.unsqueeze(0)
    input_batch = input_img_data.to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.max(probabilities, dim=0)

    print("Predicted class:", predicted_class.indices.item())


# Set up the image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Assuming the model requires 224x224 input
    transforms.ToTensor(),
])


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def generate_unique_filename(filename):
    """Generate a unique filename to avoid overwriting existing files."""
    unique_id = str(uuid.uuid4())[:8]
    return f"{secure_filename(filename)}_{unique_id}"


def query_language_identification(image_data):
    """Send the image data to the language identification model and get the response."""
    try:

        image_labels = {0: "Arabic", 1: "Chinese", 2: "Gujarati", 3: "Tamil", 4: "Thai"}

        # Load image from image data
        img = Image.open(io.BytesIO(image_data))
        img = image_transform(img).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(img)

        # Process the output, you might need to adjust this based on your model's architecture
        probabilities = F.softmax(output, dim=1).squeeze(0)
        _, predicted_class = torch.max(probabilities, dim=0)

        return {"predicted_class": image_labels[predicted_class.item()], "probabilities": probabilities.tolist()}

    except Exception as e:
        return {"error": str(e)}


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

        # Call the language identification function and get the response
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


def predict(img):
    img = image_transform(img).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        output = model(img)

    probabilities = F.softmax(output, dim=1).squeeze(0)
    _, predicted_class = torch.max(probabilities, dim=0)

    return {"predicted_class": predicted_class.item(), "probabilities": probabilities.tolist()}


inputs = gr.inputs.Image()
outputs = gr.outputs.Label(num_top_classes=3)

gr.Interface(fn=predict, inputs=inputs, outputs=outputs).launch()
