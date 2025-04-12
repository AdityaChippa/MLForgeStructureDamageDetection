import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

# Path to the folder where all the measurement data is stored
DATA_DIR = r"D:\Data"

# Different system conditions grouped into categories
STATES = {
    'baseline': ['state#13'],
    'environmental': ['state#01', 'state#02', 'state#17', 'state#18', 'state#21', 'state#22', 'state#23', 'state#24'],
    'nonlinear': ['state#08', 'state#09', 'state#10', 'state#11', 'state#12'],
    'combined': ['state#14', 'state#15', 'state#16']
}

# Map file number ranges to their corresponding system states
FILE_TO_STATE = {
    range(11, 21): 'state#01',
    range(21, 31): 'state#02',
    range(160, 170): 'state#08',
    range(170, 180): 'state#09',
    range(180, 190): 'state#10',
    range(190, 200): 'state#11',
    range(200, 210): 'state#12',
    range(210, 220): 'state#13',
    range(220, 230): 'state#14',
    range(230, 240): 'state#15',
    range(240, 250): 'state#16',
    range(251, 261): 'state#17',
    range(261, 271): 'state#18',
    range(291, 301): 'state#21',
    range(302, 312): 'state#22',
    range(312, 322): 'state#23',
    range(322, 332): 'state#24',
}

def get_state_from_file_number(file_num):
    for num_range, state in FILE_TO_STATE.items():
        if file_num in num_range:
            return state
    return None

def is_damaged(state):
    if state in STATES['nonlinear'] or state in STATES['combined']:
        return 1
    return 0

def load_data_file(file_path):
    try:
        data = np.loadtxt(file_path, dtype=float)
        if data.ndim == 2 and data.shape[1] == 5:
            return data
        else:
            print(f"Warning: Unexpected data shape in {file_path}: {data.shape}")
            return None
    except Exception as e:
        print(f"Error loading file: {file_path}, Error: {str(e)}")
        return None

def extract_features(signal_data):
    features = {}
    if signal_data is None or len(signal_data) == 0:
        return None

    channels = ["force", "accel1", "accel2", "accel3", "accel4"]
    
    for i, channel in enumerate(channels):
        channel_data = signal_data[:, i]
        features[f"{channel}_mean"] = np.mean(channel_data)
        features[f"{channel}_std"] = np.std(channel_data)
        features[f"{channel}_max"] = np.max(channel_data)
        features[f"{channel}_min"] = np.min(channel_data)
        features[f"{channel}_rms"] = np.sqrt(np.mean(channel_data**2))
        features[f"{channel}_kurtosis"] = (np.mean((channel_data - np.mean(channel_data))**4) / 
                                           (np.std(channel_data)**4))
        features[f"{channel}_skewness"] = (np.mean((channel_data - np.mean(channel_data))**3) / 
                                           (np.std(channel_data)**3))
        if len(channel_data) > 1:
            fft_vals = np.abs(np.fft.fft(channel_data))
            fft_freq = np.fft.fftfreq(len(channel_data))
            pos_mask = fft_freq > 0
            fft_vals = fft_vals[pos_mask]
            fft_freq = fft_freq[pos_mask]
            if len(fft_vals) > 3:
                top_indices = np.argsort(fft_vals)[-3:]
                for idx, j in enumerate(top_indices):
                    features[f"{channel}_dom_freq_{idx+1}"] = fft_freq[j]
                    features[f"{channel}_dom_amp_{idx+1}"] = fft_vals[j]
            features[f"{channel}_spectral_mean"] = np.mean(fft_vals)
            features[f"{channel}_spectral_std"] = np.std(fft_vals)
            features[f"{channel}_spectral_kurtosis"] = (np.mean((fft_vals - np.mean(fft_vals))**4) / 
                                                       (np.std(fft_vals)**4)) if np.std(fft_vals) > 0 else 0
    
    for i in range(1, len(channels)):
        for j in range(i+1, len(channels)):
            ch_i = signal_data[:, i]
            ch_j = signal_data[:, j]
            corr = np.corrcoef(ch_i, ch_j)[0, 1]
            features[f"corr_{channels[i]}_{channels[j]}"] = corr

    return features

# Load all files, extract features, and assign labels
def load_and_process_all_data(data_dir):
    all_features = []
    labels = []
    file_paths = []

    for state_folder in os.listdir(data_dir):
        state_path = os.path.join(data_dir, state_folder)
        if not os.path.isdir(state_path):
            continue
        for file_name in os.listdir(state_path):
            if file_name.startswith("data"):
                file_paths.append(os.path.join(state_path, file_name))
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        try:
            file_num_str = file_name.replace("data", "").split('.')[0]
            file_num = int(file_num_str)
        except:
            print(f"Couldn't parse file number from: {file_name}")
            continue

        state = get_state_from_file_number(file_num)
        if state is None:
            print(f"Unknown state for file: {file_name}")
            continue

        data = load_data_file(file_path)
        features = extract_features(data)

        if features is not None:
            all_features.append(features)
            labels.append(is_damaged(state))
    
    features_df = pd.DataFrame(all_features)
    return features_df, np.array(labels)

# Train a machine learning model using the extracted features
def build_and_train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    return best_model, X_test, y_test

# Show how the model is performing
def visualize_results(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model[-1], 'feature_importances_'):
        feature_importances = model[-1].feature_importances_
        feature_names = X_test.columns
        sorted_idx = np.argsort(feature_importances)
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

# Kick off the full analysis pipeline
def main():
    print("Starting Structural Health Monitoring ML Pipeline")
    
    print("Loading and processing data...")
    X, y = load_and_process_all_data(DATA_DIR)
    print(f"Dataset created with {X.shape[0]} samples and {X.shape[1]} features")
    
    print("Saving processed dataset...")
    X['damage_label'] = y
    X.to_csv('processed_dataset.csv', index=False)
    X = X.drop('damage_label', axis=1)
    
    print("Training model...")
    model, X_test, y_test = build_and_train_model(X, y)
    
    print("Saving trained model...")
    joblib.dump(model, 'structural_damage_model.pkl')
    
    print("Generating visualizations...")
    visualize_results(model, X_test, y_test)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
