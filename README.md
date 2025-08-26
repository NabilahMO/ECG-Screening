
# ECG Arrhythmia Classification using Machine Learning

## Project Goal
Build a machine learning model capable of **classifying individual heartbeats** from ECG signals into **Normal (N)** or one of several **AAMI-defined arrhythmia classes (F, Q, S, V)** using extracted waveform features.

## Dataset
- **Source**: Pre-processed ECG beat segments.
- **Total samples**: 109,448 beats.
- **Label distribution** (AAMI standard grouping):
  - `N`: Normal beats  
  - `S`: Supraventricular ectopic beats  
  - `V`: Ventricular ectopic beats  
  - `F`: Fusion beats  
  - `Q`: Unknown class (e.g., artifacts)

## ML Pipeline Overview

### 1. Data Loading & Preprocessing
- Load `all_records_features.csv`
- Handle missing values (column-wise mean imputation)
- Construct feature matrix `X` and target vector `y` with 8 ECG-derived features

### 2. Feature Set
- `rr_prev`, `rr_next`: RR intervals (before and after beat)
- `r_peak_amp`, `q_peak_amp`, `s_peak_amp`: Wave amplitudes
- `mean`, `std`, `skew`: Beat shape statistics

### 3. Label Grouping (AAMI)
```python
label_mapping = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    'Q': 'Q'
}
```

### 4. Train-Test Split
- Stratified 80/20 split to preserve label proportions

### 5. Model Training
- **Classifier**: `RandomForestClassifier`
- **Metrics**:
  - Accuracy: **98%**
  - Precision/Recall (macro avg): **0.96 / 0.86**
- Visuals: Confusion matrix and feature importance chart
![Confusion Matrix](./confusion_matrix.png)
![Feature Importance](./feature_importance.png)

### 6. Top Features
- `rr_prev`
- `rr_next`
- `std`
- `skew`
- `r_peak_amp`

## Key Highlights
- High performance across all arrhythmia classes
- Feature-based, interpretable model
- Potential for mobile or wearable ECG screening tools

## **Getting Started:**
### **Requirements**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### **Running the Project**

```
jupyter notebook
# Then open notebooks/01_Data_Preprocessing.ipynb and 02_ML_Training.ipynb
```

## References
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

## Acknowledgements
- PhysioNet MIT-BIH ECG Database
- scikit-learn documentation

## Designed and built by Nabilah Muri-Okunola as part of a growing portfolio exploring the intersection of AI, healthcare, and human-centred design.