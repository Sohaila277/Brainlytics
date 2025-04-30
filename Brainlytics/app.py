import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import cv2
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, static_folder='static')
CORS(app)

# Models and settings
TUMOR_INPUT_SHAPE = (299, 299)
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Load the tumor classifier model
tumor_model = load_model('brain_tumor_classifier.h5')

# Load the MRI detector model
mri_detector_model = load_model('mri_detector_model.h5')

class BrainSegmentationModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = T.Compose([T.ToTensor()])

    def _load_model(self, path):
        model = smp.Unet(
            encoder_name="efficientnet-b7",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def predict(self, image_np):
        image = Image.fromarray(image_np)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        return (output > 0.5).float().squeeze().cpu().numpy()

segmentation_model = BrainSegmentationModel("brain_segmentation_model.pth")

# Helper functions
def preprocess(image, target_shape):
    """
    Preprocess image for the model
    - Resize to target shape
    - Convert to array and normalize
    - Handle grayscale images
    """
    image = image.resize(target_shape)
    image_array = img_to_array(image) / 255.0

    if image_array.shape[-1] != 3:
        image_array = np.stack((image_array.squeeze(),) * 3, axis=-1)

    return np.expand_dims(image_array, axis=0)

# Function to predict if image is MRI or not using the MRI detector model
def is_mri_image(image):
    # Preprocess image for MRI detector
    processed = preprocess(image, (128, 128))
    prediction = mri_detector_model.predict(processed)
    
    # Return True if it's an MRI, False otherwise
    return prediction[0] < 0.5  # If prediction is less than 0.5, it's an MRI

# Routes
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Step 1: Check if the image is an MRI scan
        if not is_mri_image(img):
            return jsonify({'error': 'Uploaded image is not an MRI scan. Please upload a valid MRI scan.'}), 400

        # Step 2: Proceed with tumor classification if the image is an MRI
        processed = preprocess(img, TUMOR_INPUT_SHAPE)
        prediction = tumor_model.predict(processed)
        class_idx = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][class_idx])

        return jsonify({
            'prediction': class_names[class_idx],
            'confidence': round(confidence * 100, 2),
            'class': class_idx
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Step 1: Check if the image is an MRI scan
        if not is_mri_image(img):
            return jsonify({'error': 'Uploaded image is not an MRI scan. Please upload a valid MRI scan for segmentation.'}), 400

        # Step 2: Proceed with image segmentation if it's an MRI scan
        img_resized = img.resize((256, 256))
        img_np = np.array(img_resized)
        mask = segmentation_model.predict(img_np)
        mask_image = (mask * 255).astype(np.uint8)

        color_mask = np.zeros_like(img_np)
        color_mask[..., 0] = mask_image
        overlay = cv2.addWeighted(img_np, 0.7, color_mask, 0.3, 0)

        def encode(image_np):
            _, buffer = cv2.imencode('.png', image_np)
            return base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'original_image': encode(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)),
            'segmentation_mask': encode(mask_image),
            'overlay_image': encode(overlay),
            'message': 'Segmentation complete with overlay'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    print(f"Tumor Classifier expects input shape: {TUMOR_INPUT_SHAPE}")
    app.run(host='0.0.0.0', port=5000, debug=True)
