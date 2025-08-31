import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew
import os

def read_mit_bih_record(record_name, data_dir='mit-bih-arrhythmia-database-1.0.0'):
    #This function reads signal and annotation data from the database and organises it.
    try:
        record_path = os.path.join(data_dir, record_name) 
        print(f"Processing record: {record_name}...")
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        return record, annotation 
        """
        Returns a tuple containing: 
        wfdb.Record: The record object with signal data and metadata. 
        wfdb.Annotation: The annotation object with heartbeat labels.
        """
    except FileNotFoundError:
        print(f"Error: Record files not found for '{record_name}'. Skipping.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred for record '{record_name}': {e}. Skipping.")
        return None, None

def filter_ecg_signal(signal, fs, lowcut=0.5, highcut=45.0, order=4):
    #This function applies a band-pass filter to the ECG signal and returns it as an np.array.
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def segment_heartbeats(signal, annotations, fs):
    """
    This function segments the continuous ECG signal into individual heartbeats.
    It returns a tuple containing:
    A list of numpy arrays, where each array is a heartbeat segment.
    A list of the corresponding annotation symbols (labels) for each beat.
    A list of the sample indices of the R-peaks for the segmented beats.
    """
    before_samples = 100
    after_samples = 180

    segmented_beats = []
    beat_labels = []
    valid_r_peak_locations = []
    beat_annotation_symbols = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']

    r_peak_locations = annotations.sample
    r_peak_symbols = annotations.symbol

    for i in range(len(r_peak_locations)):
        r_peak_loc = r_peak_locations[i]
        symbol = r_peak_symbols[i]

        if symbol in beat_annotation_symbols:
            start = r_peak_loc - before_samples
            end = r_peak_loc + after_samples

            if start >= 0 and end < len(signal):
                beat = signal[start:end]
                segmented_beats.append(beat)
                beat_labels.append(symbol)
                valid_r_peak_locations.append(r_peak_loc)
                
    return segmented_beats, beat_labels, valid_r_peak_locations

def extract_features(segmented_beats, valid_r_peaks, fs):
    
    #This function extracts features from each segmented heartbeat.
    
    all_features = []
    rr_intervals = np.diff(valid_r_peaks) / fs

    for i, beat in enumerate(segmented_beats):
        rr_prev = rr_intervals[i-1] if i > 0 else np.nan
        rr_next = rr_intervals[i] if i < len(rr_intervals) else np.nan
        
        r_peak_amp = beat[100]
        q_peak_amp = np.min(beat[:100])
        s_peak_amp = np.min(beat[100:])
        
        mean_val = np.mean(beat)
        std_val = np.std(beat)
        skew_val = skew(beat)
        
        features = {
            'rr_prev': rr_prev,
            'rr_next': rr_next,
            'r_peak_amp': r_peak_amp,
            'q_peak_amp': q_peak_amp,
            's_peak_amp': s_peak_amp,
            'mean': mean_val,
            'std': std_val,
            'skew': skew_val
        }
        all_features.append(features)
        
    return all_features