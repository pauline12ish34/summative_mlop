"""
Model Module for Shoe Classification
Handles model creation, training, and retraining
Uses the same custom CNN architecture as the trained model in the notebook
"""

import os
import json
import numpy as np
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D, MaxPooling2D,
                                    BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras import regularizers


class ShoeClassifier:
    """
    Shoe classification model using custom CNN
    Architecture matches the notebook training pipeline
    """
    
    def __init__(self, img_height=128, img_width=128, num_classes=3):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['Boot', 'Sandal', 'Shoe']
        
    def build_model(self, use_pretrained=False):
        """
        Build custom CNN model architecture (same as notebook)
        
        Args:
            use_pretrained: Ignored - kept for API compatibility
        """
        # Build custom CNN matching notebook architecture
        self.model = Sequential(name='CNN_shoe_model')
        
        # First Conv Block
        self.model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(0.0005),
                             kernel_initializer=HeNormal(),
                             input_shape=(self.img_height, self.img_width, 3),
                             name='CONV_Layer1'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(BatchNormalization())
        
        # Second Conv Block
        self.model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                             kernel_initializer=HeNormal(), name='CONV_Layer2'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        
        # Third Conv Block
        self.model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                             kernel_initializer=HeNormal(), name='CONV_Layer3'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        
        # Flatten and Dense Layers
        self.model.add(Flatten(name='Flatten'))
        self.model.add(Dense(220, activation='relu', kernel_initializer=HeNormal(), name='FullyConnected1'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax', 
                           kernel_initializer=HeNormal(), name='OutputLayer'))
        
        # Compile model (same as notebook)
        self.model.compile(
            optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def train(self, train_generator, val_generator, epochs=20, patience=5):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Update class names from generator
        self.class_names = list(train_generator.class_indices.keys())
        
        return history
    
    def retrain(self, train_generator, val_generator, model_path, epochs=10):
        """
        Retrain the model using existing model as base
        
        Args:
            train_generator: Training data generator with new data
            val_generator: Validation data generator
            model_path: Path to existing model
            epochs: Number of epochs for retraining
            
        Returns:
            Training history
        """
        # Load existing model
        self.load_model(model_path)
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        # Retrain
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1
        )
        
        return history
    
    def save_model(self, save_dir, version=None):
        """
        Save the model and class names
        
        Args:
            save_dir: Directory to save model
            version: Optional version string
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if version:
            model_filename = f'shoe_classifier_model_v{version}.h5'
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f'shoe_classifier_model_{timestamp}.h5'
        
        model_path = os.path.join(save_dir, model_filename)
        self.model.save(model_path)
        
        # Save class names
        class_names_path = os.path.join(save_dir, 'class_names.json')
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f)
        
        # Save latest model path
        latest_path = os.path.join(save_dir, 'shoe_classifier_model.h5')
        self.model.save(latest_path)
        
        return model_path
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path: Path to model file
        """
        self.model = load_model(model_path)
        
        # Load class names
        model_dir = os.path.dirname(model_path)
        class_names_path = os.path.join(model_dir, 'class_names.json')
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        
        return self.model
    
    def evaluate(self, val_generator):
        """
        Evaluate the model
        
        Args:
            val_generator: Validation data generator
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")
        
        results = self.model.evaluate(val_generator, verbose=0)
        
        metrics = {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'precision': float(results[2]),
            'recall': float(results[3]),
            'auc': float(results[4])
        }
        
        # Calculate F1-Score
        precision = metrics['precision']
        recall = metrics['recall']
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics['f1_score'] = float(f1_score)
        
        return metrics
