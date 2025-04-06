from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import os
from flask_cors import CORS
from image_segmentation import find_equation

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the base64 image data from the request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400
            
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]  # Remove the "data:image/jpeg;base64," prefix
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save temporarily
        temp_path = 'temp_image.jpg'
        image.save(temp_path)
        
        # Process the image using image segmentation
        equation, result = find_equation(temp_path)
        
        # Print results to command line
        print("\nProcessing Results:")
        print(f"Equation: {equation}")
        print(f"Result: {result}")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Return both the equation and its result
        return jsonify({
            'equation': equation,
            'result': result
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Server is running on http://localhost:8000")
    app.run(debug=True, port=8000) 