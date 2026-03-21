import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import joblib
import json
import os

CSV_PATH = r"c:\Users\divya\OneDrive\Desktop\Yoga_guide\backend\yoga_dataset.csv"
MODEL_SAVE_PATH = r"c:\Users\divya\OneDrive\Desktop\Yoga_guide\backend\models"

def normalize_landmarks(X_raw):
    X_norm = np.copy(X_raw)
    for i in range(X_raw.shape[0]):
        row = X_raw[i]
        # Center: Midpoint of left hip (23) and right hip (24)
        hip_x = (row[92] + row[96]) / 2.0
        hip_y = (row[93] + row[97]) / 2.0
        hip_z = (row[94] + row[98]) / 2.0
        
        # Translate
        for j in range(33):
            X_norm[i, j*4] -= hip_x
            X_norm[i, j*4+1] -= hip_y
            X_norm[i, j*4+2] -= hip_z
            
        # Scale by maximum distance to normalize height
        coords = []
        for j in range(33):
            coords.append(np.sqrt(X_norm[i, j*4]**2 + X_norm[i, j*4+1]**2 + X_norm[i, j*4+2]**2))
        max_dist = max(coords)
        if max_dist > 0:
            for j in range(33):
                X_norm[i, j*4] /= max_dist
                X_norm[i, j*4+1] /= max_dist
                X_norm[i, j*4+2] /= max_dist
                
    return X_norm

def main():
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
        
    print(f"Loading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    df['target'] = df['pose'] + '_' + df['quality']
    
    X = df.drop(columns=['pose', 'quality', 'target']).values  # 132 features
    y_raw = df['target'].values
    
    # 1. Root-Relative Normalization
    X_centered = normalize_landmarks(X)
    
    # 2. Scale Features for 90%+ Accuracy
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_centered)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    
    num_classes = len(np.unique(y_encoded))
    print(f"Total Unique Classes to Predict: {num_classes}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("\n--- Starting Training ---")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n--- Final Test Accuracy: {accuracy * 100:.2f}% ---")

    # 8. Save Model, Labels & History
    import json
    with open(os.path.join(MODEL_SAVE_PATH, 'training_history.json'), 'w') as f:
        json.dump(history.history, f)

    model.save(os.path.join(MODEL_SAVE_PATH, 'yoga_custom_nn.keras'))
    joblib.dump(label_encoder, os.path.join(MODEL_SAVE_PATH, 'label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_SAVE_PATH, 'scaler.pkl'))
    
    print(f"\nModel saved to: {MODEL_SAVE_PATH}/yoga_custom_nn.keras")
    print(f"Labels saved to: {MODEL_SAVE_PATH}/label_encoder.pkl")
    print("Classes mapping:")
    for idx, class_name in enumerate(label_encoder.classes_):
        print(f"  {idx}: {class_name}")

if __name__ == "__main__":
    main()
