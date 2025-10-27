from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Config
IMG_SIZE = 224
NUM_CLASSES = 4
DEVICE = torch.device("cpu")

# Model definition
class TinyOcuNet(nn.Module):
    def __init__(self, num_classes):
        super(TinyOcuNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load model (must exist in project root)
model = TinyOcuNet(NUM_CLASSES)
model_path = os.path.join(os.path.dirname(__file__), '../eye_disease_model.pth')
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Class labels
class_names = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract"]

# Routes
@app.route('/', methods=['GET'])
def home():
    return render_template('../templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'No image provided'}),400

    file = request.files['file']
    img_bytes = file.read()
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except:
        return jsonify({'error':'Invalid image file'}),400

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs,1)
        pred_class = class_names[predicted.item()]

    return jsonify({'prediction':pred_class})
