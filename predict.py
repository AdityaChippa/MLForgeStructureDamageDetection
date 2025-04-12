# Structural Health Monitoring - Prediction Tool
# Author: [Your Name]
# Description: This script analyzes time-series sensor data to detect structural damage.

import os
import numpy as np
import pandas as pd
import joblib
from scipy import signal
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import time


def load_data_file(file_path):
    """Load a data file which contains time-series data."""
    try:
        data = np.loadtxt(file_path, dtype=float)

        # Expecting 5 columns: [force, accel1, accel2, accel3, accel4]
        if data.ndim == 2 and data.shape[1] == 5:
            return data
        else:
            print(f"Warning: Unexpected data shape in {file_path}: {data.shape}")
            return None
    except Exception as e:
        print(f"Error loading file: {file_path}, Error: {str(e)}")
        return None


def extract_features(signal_data):
    """Extract statistical and spectral features from time-series data."""
    features = {}

    if signal_data is None or len(signal_data) == 0:
        return None

    channels = ["force", "accel1", "accel2", "accel3", "accel4"]

    for i, channel in enumerate(channels):
        channel_data = signal_data[:, i]

        # Basic statistical features
        features[f"{channel}_mean"] = np.mean(channel_data)
        features[f"{channel}_std"] = np.std(channel_data)
        features[f"{channel}_max"] = np.max(channel_data)
        features[f"{channel}_min"] = np.min(channel_data)
        features[f"{channel}_rms"] = np.sqrt(np.mean(channel_data**2))
        features[f"{channel}_kurtosis"] = (
            np.mean((channel_data - np.mean(channel_data))**4) /
            (np.std(channel_data)**4)
        )
        features[f"{channel}_skewness"] = (
            np.mean((channel_data - np.mean(channel_data))**3) /
            (np.std(channel_data)**3)
        )

        # Frequency domain features
        if len(channel_data) > 1:
            fft_vals = np.abs(np.fft.fft(channel_data))
            fft_freq = np.fft.fftfreq(len(channel_data))

            # Use only positive frequencies
            pos_mask = fft_freq > 0
            fft_vals = fft_vals[pos_mask]
            fft_freq = fft_freq[pos_mask]

            # Top 3 dominant frequencies and their amplitudes
            if len(fft_vals) > 3:
                top_indices = np.argsort(fft_vals)[-3:]
                for idx, j in enumerate(top_indices):
                    features[f"{channel}_dom_freq_{idx+1}"] = fft_freq[j]
                    features[f"{channel}_dom_amp_{idx+1}"] = fft_vals[j]

            # Spectral statistics
            features[f"{channel}_spectral_mean"] = np.mean(fft_vals)
            features[f"{channel}_spectral_std"] = np.std(fft_vals)
            features[f"{channel}_spectral_kurtosis"] = (
                np.mean((fft_vals - np.mean(fft_vals))**4) /
                (np.std(fft_vals)**4)
            ) if np.std(fft_vals) > 0 else 0

    # Cross-correlation between accelerometer channels
    for i in range(1, len(channels)):
        for j in range(i+1, len(channels)):
            ch_i = signal_data[:, i]
            ch_j = signal_data[:, j]
            corr = np.corrcoef(ch_i, ch_j)[0, 1]
            features[f"corr_{channels[i]}_{channels[j]}"] = corr

    return features


def get_state_from_filepath(file_path):
    """Extract state name from directory structure."""
    dir_name = os.path.basename(os.path.dirname(file_path))
    if dir_name.startswith('state'):
        return dir_name
    return "unknown"


def process_and_predict(file_path, model_path):
    """Load a data file, extract features, and predict damage status."""
    data = load_data_file(file_path)
    if data is None:
        return None

    features = extract_features(data)
    if features is None:
        return None

    features_df = pd.DataFrame([features])
    model = joblib.load(model_path)

    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]

    return {
        "prediction": int(prediction),
        "status": "Damaged" if prediction == 1 else "Undamaged",
        "confidence": float(probability[prediction]),
        "state": get_state_from_filepath(file_path),
        "file": os.path.basename(file_path)
    }


def visualize_signal(file_path, output_path=None):
    """Generate and save (or show) a plot of the time-series data."""
    data = load_data_file(file_path)
    if data is None:
        print(f"Could not load data from {file_path}")
        return

    plt.figure(figsize=(14, 10))
    channels = ["Force Transducer", "Accelerometer 1", "Accelerometer 2", 
                "Accelerometer 3", "Accelerometer 4"]

    plot_len = min(1000, data.shape[0])

    for i, channel in enumerate(channels):
        plt.subplot(len(channels), 1, i+1)
        plt.plot(data[:plot_len, i])
        plt.title(channel)
        plt.grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Signal visualization saved to {output_path}")
    else:
        plt.show()


def find_data_directories(base_dir):
    """Scan and return directories that contain sensor data files."""
    data_dirs = []

    # Check if base_dir itself has data
    if any(f.startswith("data") for f in os.listdir(base_dir)):
        data_dirs.append(base_dir)

    # Walk through subdirectories
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if any(f.startswith("data") for f in os.listdir(dir_path)):
                data_dirs.append(dir_path)

    return data_dirs


def auto_process_directory(base_dir, model_path, output_csv=None):
    """Automatically process and predict all files in all subdirectories."""
    print(f"Searching for data directories in {base_dir}...")
    data_dirs = find_data_directories(base_dir)

    if not data_dirs:
        print("No data directories found!")
        return

    print(f"Found {len(data_dirs)} directories containing data files:")
    for i, dir_path in enumerate(data_dirs):
        file_count = len([f for f in os.listdir(dir_path) if f.startswith("data")])
        print(f"  {i+1}. {dir_path}: {file_count} data files")

    all_results = []
    start_time = time.time()
    file_count = 0

    print("\nProcessing all data files...")

    for dir_path in data_dirs:
        data_files = [f for f in os.listdir(dir_path) if f.startswith("data")]
        file_count += len(data_files)

        print(f"\nProcessing {len(data_files)} files in {os.path.basename(dir_path)}...")

        for file in tqdm(data_files, desc=f"{os.path.basename(dir_path)}"):
            file_path = os.path.join(dir_path, file)
            result = process_and_predict(file_path, model_path)

            if result:
                result["file_path"] = file_path
                all_results.append(result)

    end_time = time.time()
    proc_time = end_time - start_time

    if all_results:
        results_df = pd.DataFrame(all_results)

        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")

        state_summary = results_df.groupby(['state', 'status']).size().unstack(fill_value=0)

        print("\n===== Prediction Summary =====")
        print(f"Total files processed: {len(all_results)} of {file_count} "
              f"({len(all_results)/file_count*100:.1f}%)")
        print(f"Processing time: {proc_time:.2f} seconds "
              f"({proc_time/len(all_results):.4f} seconds per file)")
        print(f"Overall damage detection: {results_df['prediction'].sum()} damaged, "
              f"{len(results_df) - results_df['prediction'].sum()} undamaged")

        print("\nResults by State Condition:")
        print(state_summary)

        failed_count = file_count - len(all_results)
        if failed_count > 0:
            print(f"\nNote: {failed_count} files could not be processed due to errors.")

        return results_df

    print("No results were obtained from the data files.")
    return None


def main():
    """Entry point for command-line interface."""
    parser = argparse.ArgumentParser(description="Structural Health Monitoring Prediction Tool")

    parser.add_argument("--file", type=str, help="Path to a single data file to analyze")
    parser.add_argument("--dir", type=str, help="Directory containing data files to analyze")
    parser.add_argument("--auto", action="store_true", 
                        help="Automatically find and process all data files in the current directory")
    parser.add_argument("--model", type=str, default="structural_damage_model.pkl", 
                        help="Path to the trained model file")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize the signal data")
    parser.add_argument("--output", type=str, default="prediction_results.csv",
                        help="Output path for results (CSV for batch mode)")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return

    if args.auto:
        current_dir = os.getcwd()
        print(f"Auto mode: Processing all data files in {current_dir} and subdirectories...")
        auto_process_directory(current_dir, args.model, args.output)
        return

    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return

        print(f"Processing file: {args.file}")
        result = process_and_predict(args.file, args.model)

        if result:
            print("\nPrediction Result:")
            print(f"File: {result['file']}")
            print(f"State: {result['state']}")
            print(f"Status: {result['status']}")
            print(f"Confidence: {result['confidence']:.4f}")

        if args.visualize:
            visualize_signal(args.file, f"{os.path.splitext(args.file)[0]}_signal.png")

    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            return

        print(f"Processing all data files in: {args.dir}")
        auto_process_directory(args.dir, args.model, args.output)

    else:
        print("Please specify either --file, --dir, or --auto")
        print("\nExamples:")
        print("  Process a single file:")
        print("    python predict.py --file data/state#01/data11.txt --model structural_damage_model.pkl")
        print("\n  Process all files in a directory:")
        print("    python predict.py --dir data/state#01 --model structural_damage_model.pkl")
        print("\n  Auto process all files in current directory and subdirectories:")
        print("    python predict.py --auto --model structural_damage_model.pkl")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
