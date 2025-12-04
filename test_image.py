#test_image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

class SkinDiseaseDetector:
    def __init__(self, model_path='best_transfer_model.keras',
                 test_path="D:\\skin_detection_model\\Dataset\\Test",
                 img_size=(224, 224), batch_size=32, class_names=None):
        
        self.model_path = model_path
        self.test_path = test_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.test_generator = None
        # Class names must match the order used during training (Eczema, Non-Eczema)
        self.class_names = ['Eczema', 'Non-Eczema'] 

    def create_test_generator(self):
        """Creates a dedicated, unaugmented, unshuffled generator for the test set."""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False # CRITICAL: Must not shuffle for evaluation
        )
        # Verify the class order matches the hardcoded list
        if self.class_names != list(self.test_generator.class_indices.keys()):
            print("⚠️ WARNING: Class order mismatch detected. Ensure class_names is correct.")
            self.class_names = list(self.test_generator.class_indices.keys())
            
        print(f"Classes detected in Test folder: {self.class_names}")
        print(f"Test data size: {self.test_generator.samples} images.")
        
    def load_saved_model(self):
        """Load the best trained model file."""
        try:
            # Load model must handle custom objects if you had them, but ResNet50 is standard.
            self.model = load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model from {self.model_path}. Ensure file exists.")
            print(f"Details: {e}")
            self.model = None

    def evaluate_test_set(self):
        """Evaluates the model on the completely unseen test set and generates reports."""
        if self.model is None:
            print("Model not loaded. Cannot run evaluation.")
            return

        if self.test_generator is None:
            self.create_test_generator()
        
        print("\n--- Starting Final Evaluation on UNSEEN Test Set ---")
        
        # Calculate steps for clean evaluation
        steps = self.test_generator.samples // self.test_generator.batch_size
        if self.test_generator.samples % self.test_generator.batch_size != 0:
            steps += 1
            
        # 1. Evaluate loss and accuracy
        loss, accuracy = self.model.evaluate(self.test_generator, steps=steps, verbose=1)
        print(f"\n✨ Final Test Set Accuracy: {accuracy:.4f}")
        print(f"Final Test Set Loss: {loss:.4f}")

        # 2. Get predictions and reports
        self.test_generator.reset() 
        predictions = self.model.predict(self.test_generator, steps=steps, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes 
        
        # Truncate y_true if necessary (due to incomplete final batch predictions)
        y_true = y_true[:len(y_pred)]

        print("\nClassification Report (Test Set):")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        # 3. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix_test.png')
        plt.show() # Display the plot
        plt.close()

    def predict_single_image(self, image_path):
        """Predict the class of a single, custom image."""
        if self.model is None:
            print("Model not loaded. Cannot run prediction.")
            return None, None

        print(f"\n--- Predicting class for: {os.path.basename(image_path)} ---")
        
        try:
            # Load and convert image using PIL and NumPy (to avoid BGR issues with CV2 read)
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
            
            # Preprocessing (Resize and Scale)
            img_resized = cv2.resize(img_np, self.img_size)
            img_scaled = img_resized / 255.0
            
            # Add batch dimension
            img_final = np.expand_dims(img_scaled, axis=0)
            
            prediction = self.model.predict(img_final, verbose=0)
            predicted_index = np.argmax(prediction)
            
            predicted_class = self.class_names[predicted_index]
            confidence = np.max(prediction) * 100
            
            print(f"Predicted Class: **{predicted_class}**")
            print(f"Confidence: **{confidence:.2f}%**")
            
            # Display image in a new window/inline (for environments that support it)
            plt.figure()
            plt.imshow(img_resized)
            plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
            plt.axis('off')
            plt.show()

            return predicted_class, confidence
        
        except Exception as e:
            print(f"An error occurred during single image prediction: {e}")
            return None, None


def main():
    # --- Configuration ---
    # Path to the BEST model file saved by your training script (skin_disease_detection.py)
    MODEL_FILE = 'best_transfer_model.keras' 
    # Path to your test dataset folder
    TEST_DATA_PATH = "D:\\skin_detection_model\\Dataset\\Test"
    # Example path for a single image prediction (UPDATE THIS to a real image path)
    SINGLE_IMAGE_PATH = r"D:\skin_detection_model\Dataset\Test\Eczema\img_001.jpg" 
    
    detector = SkinDiseaseDetector(
        model_path=MODEL_FILE,
        test_path=TEST_DATA_PATH,
        img_size=(224, 224),
        batch_size=32
    )

    # 1. Load the Model
    detector.load_saved_model()

    # 2. Prepare Test Data and Run Full Evaluation
    detector.create_test_generator()
    detector.evaluate_test_set()

    # 3. Predict a Single Image 
    # Make sure to replace SINGLE_IMAGE_PATH with a valid path to test its performance 
    if os.path.exists(SINGLE_IMAGE_PATH):
        detector.predict_single_image(SINGLE_IMAGE_PATH)
    else:
        print(f"\n⚠️ Skipped single image prediction: Example image not found at {SINGLE_IMAGE_PATH}")


if __name__ == "__main__":
    main()