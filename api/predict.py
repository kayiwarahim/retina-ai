# api/predict.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io, os, requests

# =====================================================
# CONFIGURATION
# =====================================================
IMG_SIZE = 224
NUM_CLASSES = 4
DEVICE = torch.device("cpu")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../eye_disease_model.pth")
MODEL_URL = "https://huggingface.co/kayiwarahim/eye_disease_model/resolve/main/eye_disease_model.pth"

# =====================================================
# DEFINE MODEL
# =====================================================
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

# =====================================================
# DOWNLOAD MODEL IF MISSING
# =====================================================
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model from Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully!")
        else:
            raise Exception(f"❌ Failed to download model: {response.status_code}")

# =====================================================
# LOAD MODEL FUNCTION
# =====================================================
def load_model():
    download_model()
    model = TinyOcuNet(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================================
# CLASS LABELS
# =====================================================
class_names = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract"]

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_image(image_bytes, model):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        return {'error': 'Invalid image file'}

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = class_names[predicted.item()]

    return {'prediction': pred_class}
