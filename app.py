# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from api.predict import predict_image  # import the function that handles model & prediction

app = Flask(__name__)
CORS(app)  # enable CORS so frontend can call API

# ================= ROUTES =================
@app.route('/', methods=['GET'])
def home_page():
    # Serve your frontend HTML
    return render_template('index.html')  # make sure this file is in templates/

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['file']
    result = predict_image(file.read())  # ✅ only pass image bytes

    return jsonify(result)

# ================= RUN APP =================
if __name__ == '__main__':
    # host 0.0.0.0 allows external access, port 10000 can be changed if needed
    app.run(host='0.0.0.0', port=10000, debug=True)
