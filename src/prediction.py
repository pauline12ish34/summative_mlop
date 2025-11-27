"""
Prediction Module for Shoe Classification
Handles single image predictions and batch predictions
"""

import os
import json
import io
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class ShoePredictor:
    """
    Handles predictions for shoe classification
    """
    
    def __init__(self, model_path, class_names_path=None, img_height=128, img_width=128):
        """
        Initialize predictor with model
        
        Args:
            model_path: Path to trained model
            class_names_path: Path to class names JSON (optional)
            img_height: Image height for preprocessing
            img_width: Image width for preprocessing
        """
        self.model = load_model(model_path)
        self.img_height = img_height
        self.img_width = img_width
        
        # Load class names
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            # Try loading from model directory
            model_dir = os.path.dirname(model_path)
            default_path = os.path.join(model_dir, 'class_names.json')
            if os.path.exists(default_path):
                with open(default_path, 'r') as f:
                    self.class_names = json.load(f)
            else:
                self.class_names = ['Boot', 'Sandal', 'Shoe']
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict_single(self, image_path):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Create probabilities dictionary
        probabilities = {}
        for i, class_name in enumerate(self.class_names):
            probabilities[class_name] = float(predictions[0][i])
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_bytes(self, image_bytes):
        """
        Predict from image bytes (for API uploads)
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary with prediction results
        """
        # Load image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Create probabilities dictionary
        probabilities = {}
        for i, class_name in enumerate(self.class_names):
            probabilities[class_name] = float(predictions[0][i])
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python prediction.py <model_path> <image_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    predictor = ShoePredictor(model_path)
    result = predictor.predict_single(image_path)
    
    print(f"\nPrediction Results:")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nAll Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.2%}")
