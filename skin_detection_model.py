# skin_detection_model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
MODEL_OUTPUT_PATH = 'best_transfer_model.keras' 
# ---------------------

class SkinDiseaseDetector:
    """
    Core class to handle model definition, training, and loading using ResNet50 Transfer Learning.
    """
    def __init__(self, dataset_path="D:\\skin_detection_model\\Dataset\\Train", 
                 test_path="D:\\skin_detection_model\\Dataset\\Test", 
                 img_size=(224, 224), batch_size=32):
        
        self.dataset_path = dataset_path
        self.test_path = test_path
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.model = None
        # This list will be dynamically updated by the generator if the folder structure is different
        self.class_names = ['Eczema', 'Non-Eczema'] 
        
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def create_data_generators(self, validation_split=0.2):
        """
        Sets up the data generators for training and validation with augmentation.
        """
        # Data Augmentation and Preprocessing
        train_datagen = ImageDataGenerator(
            rescale=1./255, # Normalize pixel values
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split # Use a portion of the Train folder for validation
        )
        
        # Generator for training data
        self.train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Generator for validation data (no extra augmentation, just scaling)
        self.val_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Update class names based on what the generator found
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        print(f"Detected Classes: {self.class_names}")

    def build_model(self):
        """
        Builds the transfer learning model using ResNet50 pre-trained on ImageNet.
        """
        if self.train_generator is None:
            print("Error: create_data_generators must be called first.")
            return

        # 1. Load the ResNet50 base model
        base_model = ResNet50(
            weights='imagenet', 
            include_top=False, # We don't want the original classification layer
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # 2. Freeze the base layers (important for initial training stability)
        for layer in base_model.layers:
            layer.trainable = False

        # 3. Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x) # Reduce feature maps to a single vector
        
        # Add a dense block for classification
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        predictions = Dense(self.num_classes, activation='softmax')(x) # Use softmax for multi-class

        # 4. Create the full model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # 5. Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("\nModel Summary:")
        self.model.summary()

    def train_model(self, epochs=50):
        """
        Trains the model with defined callbacks.
        """
        if self.model is None or self.train_generator is None:
            print("Model or data generators not initialized. Call build_model and create_data_generators first.")
            return

        # Define Callbacks
        callbacks = [
            # Stop training if validation loss doesn't improve for 10 epochs
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1), 
            
            # Save the best model weights
            ModelCheckpoint(
                filepath=MODEL_OUTPUT_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when validation loss plateaus
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001, verbose=1)
        ]

        # Training
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        print("Training complete. Best model weights saved.")
        return history

    def test_model(self):
        """
        Evaluates the best saved model on the dedicated test set.
        """
        # Ensure the best model is loaded
        if not self.load_saved_model(MODEL_OUTPUT_PATH):
             print("Cannot run test: Best model file not found or failed to load.")
             return

        # Setup test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False # CRITICAL: Must not shuffle for proper evaluation
        )
        
        steps = self.test_generator.samples // self.test_generator.batch_size
        if self.test_generator.samples % self.test_generator.batch_size != 0:
            steps += 1
            
        print("\n--- Starting Final Evaluation on Test Set ---")
        
        # 1. Evaluate loss and accuracy
        loss, accuracy = self.model.evaluate(self.test_generator, steps=steps, verbose=1)
        print(f"\n✨ Final Test Set Accuracy: {accuracy:.4f}")
        print(f"Final Test Set Loss: {loss:.4f}")

        # 2. Get predictions for detailed report
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, steps=steps, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        y_true = y_true[:len(y_pred)] # Handle truncation if last batch was incomplete
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        # 3. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_test.png')
        print("Confusion matrix saved to confusion_matrix_test.png")
        plt.show()


    def save_model(self, model_path=MODEL_OUTPUT_PATH):
        """Saves the currently loaded model (usually the best one) to disk."""
        if self.model:
            try:
                self.model.save(model_path)
                print(f"✅ Model successfully saved to {model_path}")
            except Exception as e:
                print(f"❌ Error saving model: {e}")

    def load_saved_model(self, model_path):
        """Loads a saved model from a local path and assigns it to self.model."""
        try:
            self.model = load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model locally: {e}. File might not exist yet or is corrupted.")
            return False

# ----------------------------------------------------------------------------------
#                                 MAIN EXECUTION BLOCK
# ----------------------------------------------------------------------------------

def main():
    print("\n=======================================================")
    print("      🏥 Eczema Detection Model Training Script      ")
    print("=======================================================")
    
    # Initialize the Detector class
    detector = SkinDiseaseDetector() 
    
    # 1. Prepare Data Generators (Train/Validation Split)
    print("\n--- 1. Creating Data Generators ---")
    detector.create_data_generators()

    # 2. Build the Model (ResNet50 Transfer Learning)
    print("\n--- 2. Building Model Architecture ---")
    detector.build_model()
    
    # 3. Train the Model
    # Setting a low epoch count (e.g., 20-30) is often enough for transfer learning. 
    # The EarlyStopping callback will prevent overfitting if the number is too high.
    print(f"\n--- 3. Starting Training for up to 30 Epochs ---")
    print(f"Best model will be saved to: {MODEL_OUTPUT_PATH}")
    detector.train_model(epochs=30) 
    
    # 4. Final Evaluation on the Test Set
    # This automatically loads the best saved model (due to load_saved_model in test_model)
    print("\n--- 4. Running Final Test Set Evaluation ---")
    detector.test_model() 
    
    print("\n=======================================================")
    print("Training and Evaluation Complete.")
    print(f"The best model is saved as: {MODEL_OUTPUT_PATH}")
    print("=======================================================")


if __name__ == "__main__":
    # Ensure all TensorFlow resources are properly managed
    with tf.device('/CPU:0'): # You can change this to '/GPU:0' if you have a configured GPU
        main()