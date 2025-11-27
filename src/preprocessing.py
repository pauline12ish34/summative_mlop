"""
Data Preprocessing Module for Shoe Classification
Handles image loading, preprocessing, and data augmentation
"""

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


class DataPreprocessor:
    """
    Handles all data preprocessing operations for the shoe classifier
    """
    
    def __init__(self, img_height=128, img_width=128, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        
    def create_data_generators(self, data_dir, validation_split=0.2):
        """
        Create training and validation data generators with augmentation
        
        Args:
            data_dir: Directory containing the dataset
            validation_split: Fraction of data to use for validation
            
        Returns:
            train_generator, val_generator
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Create validation generator
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        return train_generator, val_generator
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def preprocess_uploaded_images(self, upload_dir, save_dir):
        """
        Preprocess uploaded images for retraining
        
        Args:
            upload_dir: Directory containing uploaded images
            save_dir: Directory to save preprocessed images
            
        Returns:
            Number of images processed
        """
        os.makedirs(save_dir, exist_ok=True)
        count = 0
        
        for root, dirs, files in os.walk(upload_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    try:
                        # Load and resize image
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, (self.img_width, self.img_height))
                        
                        # Save preprocessed image
                        relative_path = os.path.relpath(img_path, upload_dir)
                        save_path = os.path.join(save_dir, relative_path)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, img)
                        count += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        
        return count
