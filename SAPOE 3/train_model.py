import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define ideal conditions for 22 crops based on agricultural guidelines
# This ensures that our synthetic dataset and model reflect realistic ranges
crop_ranges = {
    'rice': {
        'N': (80, 100), 'P': (35, 50), 'K': (35, 45),
        'temp': (20.0, 27.0), 'humidity': (80.0, 85.0),
        'pH': (5.5, 6.8), 'rainfall': (180.0, 250.0)
    },
    'maize': {
        'N': (60, 80), 'P': (35, 50), 'K': (15, 25),
        'temp': (18.0, 27.0), 'humidity': (55.0, 70.0),
        'pH': (5.5, 7.0), 'rainfall': (60.0, 110.0)
    },
    'chickpea': {
        'N': (20, 40), 'P': (55, 75), 'K': (75, 85),
        'temp': (17.0, 23.0), 'humidity': (15.0, 20.0),
        'pH': (5.5, 8.5), 'rainfall': (35.0, 45.0)
    },
    'kidneybeans': {
        'N': (10, 35), 'P': (45, 60), 'K': (15, 25),
        'temp': (15.0, 25.0), 'humidity': (18.0, 25.0),
        'pH': (5.5, 6.0), 'rainfall': (60.0, 150.0)
    },
    'pigeonpeas': {
        'N': (10, 40), 'P': (65, 80), 'K': (15, 25),
        'temp': (18.0, 35.0), 'humidity': (45.0, 65.0),
        'pH': (4.5, 7.5), 'rainfall': (90.0, 200.0)
    },
    'mothbeans': {
        'N': (10, 40), 'P': (45, 60), 'K': (15, 25),
        'temp': (25.0, 32.0), 'humidity': (40.0, 65.0),
        'pH': (3.5, 10.0), 'rainfall': (30.0, 70.0)
    },
    'mungbean': {
        'N': (10, 40), 'P': (35, 50), 'K': (15, 25),
        'temp': (27.0, 30.0), 'humidity': (80.0, 90.0),
        'pH': (6.2, 7.2), 'rainfall': (35.0, 60.0)
    },
    'blackgram': {
        'N': (20, 40), 'P': (55, 70), 'K': (15, 25),
        'temp': (25.0, 35.0), 'humidity': (60.0, 70.0),
        'pH': (6.5, 7.5), 'rainfall': (60.0, 75.0)
    },
    'lentil': {
        'N': (10, 30), 'P': (55, 70), 'K': (15, 25),
        'temp': (18.0, 30.0), 'humidity': (60.0, 70.0),
        'pH': (5.5, 7.0), 'rainfall': (40.0, 50.0)
    },
    'pomegranate': {
        'N': (0, 40), 'P': (10, 30), 'K': (35, 45),
        'temp': (18.0, 25.0), 'humidity': (85.0, 95.0),
        'pH': (5.5, 7.5), 'rainfall': (100.0, 110.0)
    },
    'banana': {
        'N': (80, 120), 'P': (75, 95), 'K': (45, 55),
        'temp': (25.0, 29.0), 'humidity': (75.0, 85.0),
        'pH': (5.5, 6.5), 'rainfall': (90.0, 110.0)
    },
    'mango': {
        'N': (0, 40), 'P': (15, 35), 'K': (25, 35),
        'temp': (27.0, 36.0), 'humidity': (45.0, 55.0),
        'pH': (4.5, 7.0), 'rainfall': (90.0, 100.0)
    },
    'grapes': {
        'N': (20, 40), 'P': (120, 145), 'K': (195, 205),
        'temp': (10.0, 40.0), 'humidity': (80.0, 85.0),
        'pH': (5.5, 6.5), 'rainfall': (65.0, 75.0)
    },
    'watermelon': {
        'N': (80, 100), 'P': (5, 30), 'K': (45, 55),
        'temp': (24.0, 27.0), 'humidity': (80.0, 90.0),
        'pH': (6.0, 7.0), 'rainfall': (40.0, 60.0)
    },
    'muskmelon': {
        'N': (80, 100), 'P': (5, 30), 'K': (45, 55),
        'temp': (27.0, 30.0), 'humidity': (90.0, 95.0),
        'pH': (6.0, 6.8), 'rainfall': (20.0, 30.0)
    },
    'apple': {
        'N': (0, 40), 'P': (120, 145), 'K': (195, 205),
        'temp': (21.0, 24.0), 'humidity': (90.0, 95.0),
        'pH': (5.5, 6.5), 'rainfall': (100.0, 125.0)
    },
    'orange': {
        'N': (0, 40), 'P': (5, 30), 'K': (5, 15),
        'temp': (11.0, 35.0), 'humidity': (90.0, 95.0),
        'pH': (6.0, 8.0), 'rainfall': (100.0, 120.0)
    },
    'papaya': {
        'N': (30, 60), 'P': (45, 65), 'K': (45, 55),
        'temp': (23.0, 45.0), 'humidity': (90.0, 95.0),
        'pH': (6.5, 7.0), 'rainfall': (150.0, 250.0)
    },
    'coconut': {
        'N': (0, 40), 'P': (5, 30), 'K': (25, 35),
        'temp': (25.0, 30.0), 'humidity': (90.0, 99.0),
        'pH': (5.5, 6.5), 'rainfall': (130.0, 230.0)
    },
    'cotton': {
        'N': (100, 140), 'P': (35, 50), 'K': (15, 25),
        'temp': (22.0, 26.0), 'humidity': (75.0, 85.0),
        'pH': (5.8, 8.0), 'rainfall': (60.0, 100.0)
    },
    'jute': {
        'N': (60, 100), 'P': (35, 50), 'K': (35, 45),
        'temp': (23.0, 30.0), 'humidity': (70.0, 90.0),
        'pH': (6.0, 8.0), 'rainfall': (150.0, 200.0)
    },
    'coffee': {
        'N': (80, 120), 'P': (15, 35), 'K': (25, 35),
        'temp': (23.0, 28.0), 'humidity': (50.0, 65.0),
        'pH': (6.0, 7.5), 'rainfall': (140.0, 200.0)
    }
}

def generate_and_train(csv_path="crop_recommendation.csv", model_path="crop_model.pkl", num_samples_per_crop=100, random_seed=42):
    print("Generating synthetic crop dataset...")
    np.random.seed(random_seed)
    
    dataset_records = []
    
    for crop, limits in crop_ranges.items():
        for _ in range(num_samples_per_crop):
            record = {}
            # Generate each feature within range using a Gaussian distribution
            # Mean is the center of the range, Std Dev is 1/6th of the range size
            # This guarantees that ~99.7% of values fall within the agricultural range.
            for feature, (min_val, max_val) in limits.items():
                mean_val = (min_val + max_val) / 2.0
                std_val = (max_val - min_val) / 6.0 if max_val != min_val else 1.0
                val = np.random.normal(mean_val, std_val)
                # Clip to physical/reasonable limits
                val = np.clip(val, min_val - (std_val * 0.5), max_val + (std_val * 0.5))
                
                # Check for physical boundaries
                if feature in ['N', 'P', 'K', 'rainfall']:
                    val = max(0.0, val)
                elif feature == 'humidity':
                    val = np.clip(val, 0.0, 100.0)
                elif feature == 'pH':
                    val = np.clip(val, 3.5, 10.0)
                
                # Keep nutrients as integers and float numbers for environmental factors
                if feature in ['N', 'P', 'K']:
                    record[feature] = int(round(val))
                else:
                    record[feature] = float(np.round(val, 2))
                    
            record['label'] = crop
            dataset_records.append(record)
            
    df = pd.DataFrame(dataset_records)
    # Shuffle dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to: {csv_path} ({len(df)} rows)")
    
    # Prepare features and labels
    X = df.drop(columns=['label'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)
    
    print("Training Random Forest model (50 estimators, lightweight configuration)...")
    # Setting max_depth and min_samples_leaf ensures the model size is kept very compact
    model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=random_seed, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Calculate test accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully! Test Accuracy: {accuracy:.4f}")
    
    # Serialize model to PKL
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path} (Size: ~{os.path.getsize(model_path)/1024:.1f} KB)")
    return model

if __name__ == '__main__':
    generate_and_train()
