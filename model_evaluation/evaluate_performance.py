import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

# Paths
BASE_DIR = r"c:\Users\divya\OneDrive\Desktop\Yoga_guide\backend"
CSV_PATH = os.path.join(BASE_DIR, "yoga_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "yoga_custom_nn.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "models", "training_history.json")

def plot_learning_curves(history_data):
    """Plots training vs validation accuracy and loss over epochs."""
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history_data['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history_data['val_accuracy'], label='Validation Accuracy', color='green')
    plt.title('Epoch vs Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history_data['loss'], label='Train Error (Loss)', color='red')
    plt.plot(history_data['val_loss'], label='Validation Error (Loss)', color='orange')
    plt.title('Epoch vs Model Error (Loss)')
    plt.ylabel('Categorical Crossentropy Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Saved -> learning_curves.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plots a heatmap confusion matrix for all classes."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Yoga Pose Prediction')
    plt.ylabel('Actual Pose Class')
    plt.xlabel('Predicted Pose Class')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved -> confusion_matrix.png")
    plt.close()

def normalize_landmarks(X_raw):
    X_norm = np.copy(X_raw)
    for i in range(X_raw.shape[0]):
        row = X_raw[i]
        hip_x = (row[92] + row[96]) / 2.0
        hip_y = (row[93] + row[97]) / 2.0
        hip_z = (row[94] + row[98]) / 2.0
        for j in range(33):
            X_norm[i, j*4] -= hip_x
            X_norm[i, j*4+1] -= hip_y
            X_norm[i, j*4+2] -= hip_z
        coords = [np.sqrt(X_norm[i, j*4]**2 + X_norm[i, j*4+1]**2 + X_norm[i, j*4+2]**2) for j in range(33)]
        max_dist = max(coords)
        if max_dist > 0:
            for j in range(33):
                X_norm[i, j*4] /= max_dist
                X_norm[i, j*4+1] /= max_dist
                X_norm[i, j*4+2] /= max_dist
    return X_norm

def main():
    print("="*50)
    print("   Yoga Model Performance Evaluation   ")
    print("="*50)

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            history_data = json.load(f)
        plot_learning_curves(history_data)
    else:
        print(f"\n[Warning] {HISTORY_PATH} not found.")

    print("\nLoading dataset and model...")
    df = pd.read_csv(CSV_PATH)
    df['target'] = df['pose'] + '_' + df['quality']
    X = df.drop(columns=['pose', 'quality', 'target']).values
    y_raw = df['target'].values
    
    # Centering Data First
    X_centered = normalize_landmarks(X)
    
    # Load Scaler from 2_train_nn
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    X_scaled = scaler.transform(X_centered)
    
    label_encoder = joblib.load(ENCODER_PATH)
    y_encoded = label_encoder.transform(y_raw)
    classes = label_encoder.classes_
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\n[Accuracy & Error Values]")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Training Accuracy:  {train_acc * 100:.2f}%")
    print(f"Training Error:     {train_loss:.4f}")
    print(f"Testing Accuracy:   {test_acc * 100:.2f}%")
    print(f"Testing Error:      {test_loss:.4f}")

    print("\n[Generating Predictions for Confusion Matrix...]")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Generate Confusion Matrix image
    plot_confusion_matrix(y_test, y_pred, classes)

    # Standard Classification Report (Precision, Recall, F1)
    print("\n[Detailed Classification Report]")
    print(classification_report(y_test, y_pred, target_names=classes))

if __name__ == "__main__":
    main()
