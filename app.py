import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import io

app = Flask(__name__, static_folder='static')
CORS(app)

# Load your trained model
model = load_model('brain_tumor_model.h5')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
INPUT_SHAPE = (128, 128)  # Matching your model's expected input

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    # Resize to match model's expected input
    image = image.resize(INPUT_SHAPE)
    # Convert to numpy array
    image = img_to_array(image)
    # Ensure 3 channels (in case of grayscale)
    if image.shape[-1] != 3:
        image = np.stack((image.squeeze(),)*3, axis=-1)
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Serve frontend files
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # If user does not select file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read image file directly to memory
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # Preprocess the image
            processed_image = preprocess_image(img)
            
            # Make prediction
            prediction = model.predict(processed_image)
            
            # Get the predicted class and confidence
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
            
            # Convert to human-readable result
            if class_idx == 0:
                result = "Tumor Detected"
            else:
                result = "No Tumor Detected"
            
            return jsonify({
                'prediction': result,
                'confidence': confidence * 100,
                'class': int(class_idx)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000)