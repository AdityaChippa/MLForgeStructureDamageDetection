# MLForgeStructureDamageDetection

# 🏗️ Structural Health Monitoring System

A machine learning-based system for detecting structural damage in a three-story aluminum building structure using vibration data.  
This project was developed for the **ML Forge** competition as part of **Shilp '25**.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)  
- [Dataset Description](#dataset-description)  
- [Installation](#installation)  
- [Usage Guide](#usage-guide)  
- [Files and Directories](#files-and-directories)  
- [Results and Visualization](#results-and-visualization)  
- [Troubleshooting](#troubleshooting)  
- [Future Improvements](#future-improvements)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)

---

## 📌 Project Overview

This system helps engineers detect **structural damage** by analyzing **vibration data** from a three-story aluminum structure.

It distinguishes between:
- ✅ *Normal environmental variations* (e.g., temperature changes, aging)
- ❌ *Actual structural damage*

### 🏗️ Test Environment Simulation

- Nonlinear damage: Introduced via suspended column and adjustable bumper  
- Environmental effects: Simulated using added mass or reduced stiffness

### 💡 Why Our Solution Stands Out

- Extracts meaningful patterns from complex vibration data  
- Achieves **97% accuracy** using a **Random Forest** model  
- Easy to analyze new structural data  
- Includes intuitive visualizations for clear insights

---

## 📊 Dataset Description

The dataset includes vibration readings from a specially built aluminum structure.

### 🔧 What’s Measured?

- **Force Transducer**: Measures input force  
- **Four Accelerometers**: Measure response on each floor

### 🏷️ Conditions Simulated

- Normal Baseline:  
  - Reference state (e.g., `state#13`)

- Environmental Changes (Undamaged):
  - Added weight
  - Reduced stiffness

- Damage Conditions:
  - Varying bumper-column gaps for non-linear behavior

- Mixed Conditions:
  - Combine environmental + damage to test model robustness

Each condition contains **multiple files** with full sensor readings.

---

## ⚙️ Installation

### 📥 Clone the Repository

```bash
git clone https://github.com/your-username/structural-health-monitoring.git
cd structural-health-monitoring
```

### 🐍 Set Up a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 📦 Install Required Packages

```bash
pip install numpy pandas matplotlib scikit-learn scipy joblib seaborn tqdm
```

Or use a `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 🧠 Train the Model

```bash
python structural_health_monitoring.py
```

What it does:
- Loads and processes vibration data  
- Extracts features  
- Trains a Random Forest model  
- Evaluates model and generates visualizations  
- Saves trained model & processed data

You'll see:

```
Dataset created with 169 samples and 86 features
Test accuracy: 0.9706
Confusion Matrix:
[[18  0]
 [ 1 15]]
```

---

### 🔎 Make Predictions

#### 📂 Analyze All Data Automatically

```bash
python predict.py --auto --model structural_damage_model.pkl
```

#### 🎯 Analyze a Specific Folder

```bash
python predict.py --dir "path/to/data/directory" --model structural_damage_model.pkl
```

#### 📄 Analyze a Single File (with visualization)

```bash
python predict.py --file "path/to/data/file.txt" --model structural_damage_model.pkl --visualize
```

#### 🆘 Get Help

```bash
python predict.py --help
```

---

### 📈 Prediction Output Example

```
===== Prediction Summary =====
Total files processed: 169
Overall damage detection: 50 damaged, 119 undamaged
Results saved to prediction_results.csv
```

---

## 🔬 Feature Analysis

### 🧠 What is the model "thinking"?

#### 🔥 Most Important Features

```bash
python analyze_features.py --importance --model structural_damage_model.pkl --dataset processed_dataset.csv
```

#### 🧩 Feature Clustering (PCA/TSNE)

```bash
python analyze_features.py --clustering --dataset processed_dataset.csv
```

#### 🔗 Feature Correlations

```bash
python analyze_features.py --correlations --dataset processed_dataset.csv
```

#### 🚦 Damaged vs. Undamaged Separability

```bash
python analyze_features.py --separability --dataset processed_dataset.csv
```

---

## 📁 Files and Directories

| File | Description |
|------|-------------|
| `structural_health_monitoring.py` | Core pipeline for training and processing |
| `predict.py` | Main script to analyze new data |
| `analyze_features.py` | Explains model behavior and feature interactions |
| `debug_data_loading.py` | Helps troubleshoot dataset issues |
| `structural_damage_model.pkl` | Trained model file |
| `processed_dataset.csv` | Feature-rich, clean data |
| `feature_importance.png` | Visualization of important features |
| `confusion_matrix.png` | Model performance summary |
| `requirements.txt` | Python dependency list |

---

## 📊 Results and Visualization

### ✅ Accuracy: 97.06%
- 100% precision for undamaged states
- 93.8% precision for damaged states
- Only 1 misclassification in 34 test samples

### 🖼️ Visual Insights:
- Feature Importance Plots  
- Confusion Matrix  
- PCA & t-SNE Embeddings  
- Correlation Heatmaps  

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| File not found | Check paths and extensions (`.txt`) |
| Model loading issues | Ensure model file exists and matches scikit-learn version |
| Memory errors | Try smaller file batches |
| Unknown errors | Run `debug_data_loading.py` |

---

## 🚧 Future Improvements

- Implement deep learning (CNNs, LSTMs)  
- Real-time health monitoring  
- Web dashboard for live visualization  
- Damage localization (multi-class classification)  
- Adaptation for different structure types

---

## 📄 License

This project is under the **MIT License**. See the `LICENSE` file for more details.

---

## 🙏 Acknowledgments

Special thanks to:
- Shilp '25 / ML Forge Team – for organizing the challenge  
- Open-source contributors – for the amazing tools and libraries  
