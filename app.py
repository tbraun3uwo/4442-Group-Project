from flask import Flask, request, jsonify
from fastai.learner import load_learner
import base64
import io
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model when the server starts
model = None
try:
    model = load_learner('model/model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

def detector(digImg):
    prediction = model.predict(digImg)[0]
    if prediction in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
        return str(prediction)  # Keep as string for JSON serialization
    elif prediction in ["x", "y", "z"]:
        return prediction
    else:
        match prediction:
            case "add":
                return "+"
            case "dec":
                return "-"
            case "div":
                return "/"
            case "eq":
                return "="
            case "mul":
                return "*"
            case "sub":
                return "-"
    return prediction

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the base64 image data from the request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove the "data:image/jpeg;base64," prefix
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save temporarily if needed
        temp_path = 'temp_image.jpg'
        image.save(temp_path)
        
        # Make prediction using detector function
        pred_class, pred_idx, probs = model.predict(temp_path)
        symbol = detector(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Return prediction
        return jsonify({
            'symbol': symbol,
            'confidence': float(probs[pred_idx])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000) 