# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from api.predict import load_model, predict_image  # ✅ import from api folder

# =====================================================
# FLASK APP SETUP
# =====================================================
app = Flask(__name__)
CORS(app)

# Load model once
model = load_model()

# =====================================================
# ROUTES
# =====================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['file']
    result = predict_image(file.read(), model)

    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

# =====================================================
# RUN LOCALLY
# =====================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
