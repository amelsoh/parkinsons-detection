import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # To handle Cross-Origin Resource Sharing (important for local development)
import joblib
import cv2
import numpy as np

# --- Configuration ---
app = Flask(__name__, static_folder='static')
CORS(app) # Enable CORS for all routes (important for local development)

MODELS_DIR = 'models' # Directory where your saved models/transformers are
BEST_MODEL_FILENAME = 'randomforest_model.joblib' # Change this to your actual best model filename
MINMAX_SCALER_FILENAME = 'minmax_scaler.joblib'
PCA_TRANSFORMER_FILENAME = 'pca_transformer.joblib'
IMAGE_SIZE = 224 # Must match your training IMAGE_SIZE

# --- Load Trained Model and Transformers ---
try:
    model = joblib.load(os.path.join(MODELS_DIR, BEST_MODEL_FILENAME))
    minmax_scaler = joblib.load(os.path.join(MODELS_DIR, MINMAX_SCALER_FILENAME))
    pca_transformer = joblib.load(os.path.join(MODELS_DIR, PCA_TRANSFORMER_FILENAME))
    print("Model and transformers loaded successfully!")
except Exception as e:
    print(f"Error loading model or transformers: {e}")
    model = None
    minmax_scaler = None
    pca_transformer = None

# --- Preprocessing Functions (EXACTLY AS IN YOUR TRAINING SCRIPT) ---
# You MUST copy these functions here or import them from a module.
def grayscal_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise_image_Gaussian(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def gamma_correction(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def Otsu_threshold(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def bounding_box_extraction(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Return a black image if no contours (e.g., completely black or white image)
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def resize_image_func(image):
    return cv2.resize(image,(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

def preprocess_image_for_inference(image_data): # Takes raw image data (bytes or numpy array)
    try:
        # Assuming image_data is bytes read from request.files['file'].read()
        # Decode image data using OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Use IMREAD_COLOR to ensure 3 channels for grayscaling

        if img is None:
            raise ValueError("Could not decode image data.")

        gray_img = grayscal_image(img)
        denoised_img = denoise_image_Gaussian(gray_img)
        enhanced_img = gamma_correction(denoised_img)
        thresh_img = Otsu_threshold(enhanced_img)
        roi_img = bounding_box_extraction(thresh_img)
        processed_img = resize_image_func(roi_img)
        
        # Normalize pixel values to 0-1 range (must match training)
        processed_img = processed_img.astype('float32') / 255.0
        
        # Reshape for a single image: (1, H, W) -> (1, H*W)
        # This is (1, IMAGE_SIZE, IMAGE_SIZE) from preprocess_image_for_inference output
        # Then flatten to (1, IMAGE_SIZE*IMAGE_SIZE)
        flattened_img = processed_img.reshape(1, -1) 
        
        return flattened_img
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not minmax_scaler or not pca_transformer:
        return jsonify({'error': 'Model or transformers not loaded. Check backend logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            image_data = file.read()
            processed_image = preprocess_image_for_inference(image_data)

            if processed_image is None:
                return jsonify({'error': 'Image preprocessing failed.'}), 400
            
            # Apply scaling and PCA (must match training pipeline)
            scaled_image = minmax_scaler.transform(processed_image)
            pca_reduced_image = pca_transformer.transform(scaled_image)

            # Make prediction
            prediction = model.predict(pca_reduced_image)[0]
            probabilities = model.predict_proba(pca_reduced_image)[0]

            response = {
                'prediction': int(prediction),
                'label': 'Parkinson' if prediction == 1 else 'Healthy',
                'probabilities': {
                    'Healthy': float(probabilities[0]),
                    'Parkinson': float(probabilities[1])
                }
            }
            return jsonify(response)

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    return jsonify({'error': 'Something went wrong.'}), 500

# --- Run the Flask app ---
if __name__ == '__main__':
    # For development, run on localhost:5000
    # In production, use a WSGI server like Gunicorn
    app.run(debug=True) # debug=True auto-reloads on code changes, remove for production