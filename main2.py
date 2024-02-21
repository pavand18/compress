import numpy as np
import pandas as pd
from scipy.fftpack import dct
from sklearn.decomposition import PCA

# Read the PMU data from a CSV file (replace 'sample_data.csv' with your actual file path)
data = pd.read_csv('sample_data.csv')

# Define disturbance detection thresholds
threshold_v = 1.0  # Voltage threshold
threshold_f = 59.9  # Frequency threshold
threshold_df_dt = 0.1  # Rate of change of frequency threshold

# Function to detect disturbances
def detect_disturbance(data, threshold_v, threshold_f, threshold_df_dt):
    voltage_columns = [col for col in data.columns if 'V pu' in col]
    frequency_columns = [col for col in data.columns if 'Frequency' in col]

    disturbance_indices = []
    for v_col, f_col in zip(voltage_columns, frequency_columns):
        voltage = data[v_col]
        frequency = data[f_col]
        df_dt = np.gradient(frequency)
        
        disturbance_indices.extend(
            data.index[
                (voltage >= threshold_v) |
                (frequency < threshold_f) |
                (df_dt > threshold_df_dt)
            ].tolist()
        )
    
    return disturbance_indices

# Detect disturbances
disturbance_indices = detect_disturbance(data, threshold_v, threshold_f, threshold_df_dt)

# Determine the window length
def determine_window_length(disturbance_indices):
    if disturbance_indices:
        window_length = max(disturbance_indices) - min(disturbance_indices)
    else:
        window_length = 10  # Default window length for normal conditions
    return window_length

window_length = determine_window_length(disturbance_indices)

# Apply PCA
def apply_pca(data):
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(data)
    return pca_result

# Apply DCT compression
def apply_dct_compression(data, threshold):
    compressed_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        signal = data[:, i]
        dct_coefficients = dct(signal, type=2, norm='ortho')
        significant_indices = np.where(np.abs(dct_coefficients) > threshold)[0]
        compressed_signal = np.zeros_like(signal)
        compressed_signal[significant_indices] = dct_coefficients[significant_indices]
        compressed_data[:, i] = compressed_signal
    return compressed_data

# Prepare data for PCA
data_values = data.drop(columns=['Time']).values

# Apply PCA
pca_result = apply_pca(data_values)

# Apply DCT compression
dct_threshold = 0.1
compressed_pmu_data = apply_dct_compression(pca_result, dct_threshold)

# Your compressed PMU data is now in 'compressed_pmu_data'
# You can further process or save it as needed
