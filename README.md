# **ECG Heartbeat Classification Using Machine Learning** (Scikit-Learn)

This project utilises the Random Forest Classifier (a machine learning model) to classify individual heartbeats from ECG signals into either normal, supraventricular ectopic, ventricular ectopic or fusion beats. 

Pipeline:

1. **Data Acquisition and Feature Engineering:** 
- The `read_mit_bih_record` function reads the raw `.dat` signal (ECG) and the corresponding `.atr` annotation (the heartbeat label).
- The `filter_ecg_signal` function applies a band-pass filter to remove baseline wander and high-frequency noise.
- The `segment_heartbeats` function uses the R-peak locations from the annotations to segment the continuous signal into individual heartbeats.
- For each segmented heartbeat, a set of descriptive features are calculated using the `extract_features` function. *This is where domain knowledge is showcased.
    - *R-R intervals:* `rr_prev` and `rr_next`. The time between consecutive heartbeats.
    - *Morphological features:* `r_peak_amp`, `q_peak_amp`, `s_peak_amp`. The height and width of different parts of the heartbeat waveform.
    - *Statistical features:* `mean`, `std`, and `skew`. The mean, standard deviation, and skewness of the signal within the heartbeat window.

2. **Model Training and Evaluation:**
- The `ecg_data_prep` script creates the feature matrix `X` and target vector `y`, then splits them into `X_train`, `X_test`, `y_train`, and `y_test`.
- The `RandomForestClassifier` is trained to handle the imbalance of the heartbeat classes:
    - `stratify=y`: Uses a stratified train-test split to ensure that the class distribution in the training and testing sets is representative of the overall dataset.
    - `class_weight='balanced'`: instructs the Random Forest algorithm to pay more attention to the minority classes during training, preventing it from simply ignoring                them in favour of the overwhelmingly common 'Normal' class.
- The  `classification_report` and `ConfusionMatrixDisplay` is used to evaluate its performance with metrics like accuracy, precision, recall, and the F1-score, and create a confusion matrix to visualise which types of heartbeats it struggles with.

Built by Nabilah Muri-Okunola
